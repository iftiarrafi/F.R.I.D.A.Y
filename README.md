# ✳ F.R.I.D.A.Y — Agentic AI Assistant & Document Intelligence

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/LangGraph-1.2.0-orange?logo=langchain&logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Streamlit-1.58.0-red?logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Groq-Fast%20Inference-green?logo=lightning&logoColor=white" alt="Groq" />
  <img src="https://img.shields.io/badge/Vector%20Store-FAISS-blueviolet" alt="FAISS" />
  <img src="https://img.shields.io/badge/Embeddings-Gemini-4285F4?logo=google&logoColor=white" alt="Google Gemini" />
</p>

**F.R.I.D.A.Y** is an intelligent, multi-tool autonomous AI assistant engineered with **LangGraph**, **Groq LLM inference**, and **Streamlit**. It integrates Retrieval-Augmented Generation (RAG) over uploaded documents, live web search, real-time stock quotes, global weather telemetry, mathematical calculation, and persistent conversation memory across independent chat threads.

---

## 🎥 Video Demonstration

Click the preview below to watch the full walkthrough and live demonstration on YouTube:

<p align="center">
  <a href="https://youtu.be/xmeE2yC3Qz4" target="_blank" rel="noopener noreferrer">
    <img src="https://img.youtube.com/vi/xmeE2yC3Qz4/maxresdefault.jpg" alt="F.R.I.D.A.Y Live Demo Video" width="90%" style="border-radius: 12px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);" />
  </a>
  <br />
  <a href="https://youtu.be/xmeE2yC3Qz4" target="_blank" rel="noopener noreferrer">
    <b>▶️ Watch the Demo on YouTube: https://youtu.be/xmeE2yC3Qz4</b>
  </a>
</p>

<details>
<summary><b>Click to expand Markdown / HTML embed code</b></summary>

```markdown
<!-- Standard Markdown Image Link -->
[![F.R.I.D.A.Y Demo](https://img.youtube.com/vi/xmeE2yC3Qz4/maxresdefault.jpg)](https://youtu.be/xmeE2yC3Qz4)

<!-- Styled HTML Image Link -->
<p align="center">
  <a href="https://youtu.be/xmeE2yC3Qz4" target="_blank">
    <img src="https://img.youtube.com/vi/xmeE2yC3Qz4/maxresdefault.jpg" alt="F.R.I.D.A.Y Demo Video" width="85%" style="border-radius: 10px;">
  </a>
</p>
```

</details>

---

## ✨ Key Features

- **🧠 Autonomous Agentic Reasoning (ReAct Pattern)**: Powered by `LangGraph` and `ChatGroq`, dynamically evaluating whether to answer directly or select and execute the optimal specialized tool.
- **📄 Document RAG with PDF Upload**: 
  - Drag-and-drop PDF upload directly in the chat interface.
  - Text extraction via `PyPDFLoader` and overlapping chunking using `RecursiveCharacterTextSplitter`.
  - Dense semantic vector embeddings powered by Google's `gemini-embedding-2-preview`.
  - Local vector similarity search using `FAISS` with citation formatting.
- **🌐 Real-Time Web Search**: Advanced internet queries via `TavilySearch` for up-to-date events, breaking news, and online facts.
- **📈 Live Stock Market Quotes**: Instant financial price lookups for tickers (e.g., `AAPL`, `TSLA`, `MSFT`) via Alpha Vantage API.
- **🌤️ Global Meteorological Intelligence**: Real-time two-stage weather search (Geocoding API to resolve coordinates + Weather API) via OpenWeatherMap, reporting temperature, feels-like, humidity, wind speed, atmospheric pressure, and visibility.
- **🧮 Secure Mathematical Computation**: Safe math expression evaluator supporting trigonometry, roots, rounding, and arbitrary formulas.
- **💾 Persistent Multi-Session Memory**: SQLite checkpointer (`SqliteSaver`) preserves past chat sessions, conversation threads, and context states even after server restarts.
- **🎨 Sleek, Modern User Interface**: Clean Streamlit UI with conversation sidebar, live tool call status chips (`st.status`), and smooth token streaming.

---

## 🏗️ Architecture & Workflow

```mermaid
flowchart TD
    User([User Prompt / PDF Upload]) --> Streamlit[Streamlit UI - app.py]
    
    subgraph Document Ingestion Pipeline
        Streamlit -->|PDF File| Ingest[tools.py: ingest_rag_document]
        Ingest --> Splitter[Recursive Character Text Splitter]
        Splitter --> Embeddings[Google Gemini Embeddings]
        Embeddings --> FAISS[(FAISS Vector Database)]
    end

    subgraph LangGraph ReAct Cycle
        Streamlit -->|Chat Input + Thread ID| ChatNode[LLM Agent Node - ChatGroq]
        ChatNode --> ToolsCondition{Tool Call Needed?}
        
        ToolsCondition -->|Yes| ToolNode[Tool Execution Node]
        ToolNode -->|rag_tool| FAISS
        ToolNode -->|search_tool| Tavily[Tavily Search API]
        ToolNode -->|get_stock_price| AlphaVantage[Alpha Vantage API]
        ToolNode -->|get_current_weather| OpenWeather[OpenWeather API]
        ToolNode -->|calculator| MathEngine[Python Math Evaluator]
        
        ToolNode -->|Return Observation| ChatNode
        ToolsCondition -->|No - Final Answer| EndNode([Stream Response to UI])
    end

    subgraph State & Persistence
        ChatNode <--> SQLite[(SQLite Checkpointer - chatbot.db)]
    end
```

---

## 📁 Project Structure

```text
F.R.I.D.A.Y/
├── app.py                  # Main Streamlit frontend & chat interface
├── app_prev.py             # Previous frontend implementation backup
├── backend.py              # LangGraph state machine, nodes, and SQLite checkpointer
├── tools.py                # Tool definitions (RAG, Tavily, Weather, Stocks, Calculator)
├── requirements.txt        # Python package dependencies
├── .env.example            # Template for environment configuration
├── .gitignore              # Files and folders ignored by Git
├── DB/                     # Directory housing persistent SQLite chatbot state
│   └── chatbot.db
└── faiss_db/               # Directory housing FAISS vector database indices
    ├── index.faiss
    └── index.pkl
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/iftiarrafi/F.R.I.D.A.Y.git
cd F.R.I.D.A.Y
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Configuration

Create a `.env` file in the root directory by copying `.env.example`:

```bash
cp .env.example .env
```

Populate the keys in `.env` with your API credentials:

```dotenv
# LLM Inference (Groq)
GROQ_API_KEY="your_groq_api_key_here"

# Web Search (Tavily)
TAVLIY_API_KEY="your_tavily_api_key_here"

# Stock Quotes (Alpha Vantage)
ALPHA_VANTAGE_API_KEY="your_alpha_vantage_api_key_here"

# Real-time Weather (OpenWeatherMap)
OPENWEATHER_API_KEY="your_openweather_api_key_here"

# Gemini Embeddings (Google AI Studio)
GOOGLE_API_KEY="your_google_ai_studio_api_key_here"

# Optional: LangSmith Tracing & Telemetry
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="your_langsmith_api_key_here"
LANGSMITH_PROJECT="friday-agent"
```

> [!NOTE]
> Ensure `TAVLIY_API_KEY` matches the exact spelling in `.env` as referenced by the Tavily search tool in `tools.py`.

---

## 🚀 Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

Once running, navigate to the local URL displayed in your terminal (typically `http://localhost:8501`).

---

## 💬 Example Queries to Try

| Category | Example Prompt | Active Tool |
| :--- | :--- | :--- |
| **Document Q&A** | *Upload a research paper or PDF and ask:* `"Summarize the key findings in section 3."` | `rag_tool` |
| **Web Research** | `"What are the latest developments in generative AI announced this week?"` | `search_tool` (Tavily) |
| **Financials** | `"What is the current stock price and trading volume for NVDA?"` | `get_stock_price` |
| **Weather** | `"What is the weather like in Tokyo right now?"` | `get_current_weather` |
| **Calculations** | `"Calculate math.sqrt(144) * 15 + round(3.14159, 2)"` | `calculator` |
| **General Chat** | `"Can you explain the difference between zero-shot and few-shot prompting?"` | Direct LLM Response |

---

## 🛠️ Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) | Cyclic state graph workflow & tool binding |
| **LLM Inference** | [ChatGroq](https://groq.com/) | Ultra-low latency open-weights models (`openai/gpt-oss-120b`) |
| **Embeddings** | [Google Generative AI](https://ai.google.dev/) | `gemini-embedding-2-preview` semantic vector embeddings |
| **Vector Index** | [FAISS (Facebook AI)](https://github.com/facebookresearch/faiss) | High-speed similarity search for retrieved chunks |
| **Web UI** | [Streamlit](https://streamlit.io/) | Interactive chat application with streaming responses |
| **Storage / State** | SQLite (`SqliteSaver`) | Checkpointed multi-thread message history |
| **Live APIs** | Tavily, Alpha Vantage, OpenWeather | Real-time external data sources |

---
## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

