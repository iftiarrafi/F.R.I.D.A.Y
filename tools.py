import requests
import math
import os
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from typing import Any

from dotenv import load_dotenv
load_dotenv()


embeddings = GoogleGenerativeAIEmbeddings(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-embedding-2-preview"
)

#------ RAG Tool -------
def ingest_rag_document(file_path):
    DB_PATH = "faiss_db"
    
    loader = PyPDFLoader(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    
    vector_store = FAISS.from_documents(chunks , embeddings)
    vector_store.save_local(DB_PATH)
    
def get_retriever():
    DB_PATH = "faiss_db"

    if not (
        os.path.exists(os.path.join(DB_PATH, "index.faiss"))
        and os.path.exists(os.path.join(DB_PATH, "index.pkl"))
    ):
        return None

    vector_store = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from an uploaded PDF document.
    Use this tool for questions about PDF content.
    """
    try:
        retriever = get_retriever()

        if retriever is None:
            return (
                "No PDF has been indexed yet. "
                "Please upload a PDF before asking document questions."
            )

        documents = retriever.invoke(query)

        if not documents:
            return "No relevant information was found in the PDF."

        formatted_documents = []

        for index, document in enumerate(documents, start=1):
            page = document.metadata.get("page", "unknown")

            formatted_documents.append(
                f"Document {index}\n"
                f"Page: {page}\n"
                f"Content: {document.page_content}"
            )

        return "\n\n".join(formatted_documents)

    except Exception as error:
        return f"PDF retrieval failed: {error}"
    
#------ RAG Tool -------

search_tool = TavilySearch(
    tavily_api_key = os.getenv('TAVLIY_API_KEY'),
    max_results = 5 , topic = "general" , search_depth ="advanced"
)

@tool
def calculator(expression : str) -> str :
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def get_stock_price(symbol : str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()

@tool
def get_current_weather(location: str) -> str:
    """
    Get the current real-time weather for a given city or location.

    Args:
        location: City or location name, for example:
                "Dhaka", "London, UK", or "New York, US".

    Returns:
        A formatted current weather report.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return (
            "Weather API key is missing. "
            "Set the OPENWEATHER_API_KEY environment variable."
        )

    try:
        # Step 1: Convert the location name into latitude and longitude
        geocoding_url = "https://api.openweathermap.org/geo/1.0/direct"

        geocoding_params = {
            "q": location,
            "limit": 1,
            "appid": api_key,
        }

        geo_response = requests.get(
            geocoding_url,
            params=geocoding_params,
            timeout=10,
        )
        geo_response.raise_for_status()

        locations: list[dict[str, Any]] = geo_response.json()

        if not locations:
            return f"Could not find the location: {location}"

        latitude = locations[0]["lat"]
        longitude = locations[0]["lon"]
        resolved_name = locations[0].get("name", location)
        country = locations[0].get("country", "")
        state = locations[0].get("state", "")

        # Step 2: Get current weather using latitude and longitude
        weather_url = "https://api.openweathermap.org/data/2.5/weather"

        weather_params = {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric",
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10,
        )
        weather_response.raise_for_status()

        weather_data = weather_response.json()

        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        pressure = weather_data["main"]["pressure"]
        description = weather_data["weather"][0]["description"]
        wind_speed = weather_data.get("wind", {}).get("speed", "N/A")
        visibility_meters = weather_data.get("visibility")

        visibility_km = (
            round(visibility_meters / 1000, 1)
            if visibility_meters is not None
            else "N/A"
        )

        location_parts = [resolved_name]

        if state:
            location_parts.append(state)

        if country:
            location_parts.append(country)

        display_location = ", ".join(location_parts)

        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {description.title()}\n"
            f"- Temperature: {temperature}°C\n"
            f"- Feels like: {feels_like}°C\n"
            f"- Humidity: {humidity}%\n"
            f"- Pressure: {pressure} hPa\n"
            f"- Wind speed: {wind_speed} m/s\n"
            f"- Visibility: {visibility_km} km"
        )

    except requests.Timeout:
        return "The weather service request timed out. Please try again."

    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response else "unknown"

        if status_code == 401:
            return "The OpenWeather API key is invalid or inactive."

        return f"Weather API returned an HTTP error: {status_code}"

    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"

    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"