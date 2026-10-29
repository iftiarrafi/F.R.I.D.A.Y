import os
from tools import search_tool , get_current_weather , get_stock_price , calculator , rag_tool
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Any
from langgraph.graph import StateGraph , START , END
from typing import Annotated , TypedDict
from langchain_core.messages import BaseMessage , HumanMessage , SystemMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()


os.makedirs("DB", exist_ok=True)
conn = sqlite3.connect(database="DB/chatbot.db" , check_same_thread=False)
checkpoint = SqliteSaver(conn)



llm = ChatGroq(
    model="openai/gpt-oss-120b" ,
    api_key=os.getenv('GROQ_API_KEY') ,
)

tools = [search_tool , get_current_weather , get_stock_price , calculator , rag_tool]

tool_node = ToolNode(tools=tools)

llm_with_tools = llm.bind_tools(tools=tools)


class ChatState(TypedDict) :
    messages : Annotated[list[BaseMessage] , add_messages]
    

def chat_node(state : ChatState):
    """LLM node that can answer directly or call an appropriate tool."""

    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"

            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded PDF or document. "
            "Always retrieve relevant document content before answering PDF-related questions.\n"
            "- Use `search_tool` for current events, recent information, or information "
            "that requires an internet search.\n"
            "- Use `calculator` for mathematical calculations. Do not calculate complex "
            "expressions manually when the calculator is available.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
            "- Use `get_current_weather` when the user asks about current weather for a location.\n\n"

            "Answer general questions directly when no tool is required. "
            "Do not invent information from the uploaded document. "
            "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
            "After receiving a tool result, provide a clear and helpful final answer."
        )
    )
    
    messages = [
        system_message , *state['messages']
    ]
    
    response = llm_with_tools.invoke(messages)
    
    return {
        'messages' : [response]
    }

graph = StateGraph(ChatState)
graph.add_node("chat_node" , chat_node)
graph.add_node("tools" , tool_node)

graph.add_edge(START , "chat_node")
graph.add_conditional_edges("chat_node" , tools_condition)
graph.add_edge("tools" , "chat_node")

chatbot = graph.compile(checkpointer=checkpoint)


def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)