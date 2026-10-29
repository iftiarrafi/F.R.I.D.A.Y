import os
import sys
from backend import chatbot
from backend import get_all_threads
from langchain.messages import HumanMessage , AIMessage , ToolMessage
from tools import ingest_rag_document
import streamlit as st
import uuid
import tempfile

def generate_thread_id():
    
    return str(uuid.uuid4())


def add_thread(thread_id):
    
    if thread_id not in st.session_state["chat_threads"] :
        
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id) :
    
    state = chatbot.get_state(
        config= {
            "configurable" : {
                "thread_id" : thread_id
            }
        }
    )
    
    if not state or not state.values : return []
    
    return state.values.get("messages" , [])


def reset_chat() :
    
    st.session_state['message_history'] = []
    
    new_thread_id = generate_thread_id()
    
    st.session_state["thread_id"] = new_thread_id
    
    add_thread(new_thread_id)
    
def convert_messages(messages):
        temp = []
        
        for message in messages :
            
            if isinstance(message , HumanMessage) :
                role = "user"
            
            elif isinstance(message , AIMessage) :
                role = "assistant"
            else : continue
            
            content = message.content 
        
            temp.append({
                "role" : role , "content" : content
            })    

        return temp


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
    

add_thread(st.session_state["thread_id"])

st.title("Rafi GPT") 


    
    
# Only Sidebar interface (starts)
st.sidebar.title("Start New")

if st.sidebar.button("New Chat" ,use_container_width=True) :
    
    reset_chat()
    
    st.rerun()


# showing all chat-threads id in sidebar
st.sidebar.text("Recents")

for thread_id in reversed(st.session_state["chat_threads"]):

    messages = load_conversation(thread_id)

    # Default title for an empty conversation
    text = "Current"

    for message in messages:
        if isinstance(message, HumanMessage):
            length = len(message.content)
            if length > 30 :
                text = message.content[:30]
            else : text = message.content
            break

    if st.sidebar.button(
        text,
        key=thread_id,
        use_container_width=True
    ):
        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        st.session_state["message_history"] = convert_messages(messages)

        st.rerun()

# Only Sidebar interface (END)

# ========================= Main chat interface =========================

for message in st.session_state["message_history"] :
    
    with st.chat_message(message['role']) :
        st.text(message['content'])
        




submission = st.chat_input(
    "Type here",
    accept_file=True,
    file_type=["pdf"]
)


# Default user input value
user_input = None


# Process the submitted text and PDF
if submission:

    # Get the text entered by the user
    user_input = submission.text

    # Get the uploaded files
    # This is always a list when accept_file is enabled
    uploaded_files = submission.files

    # Process the uploaded PDF if one was attached
    if uploaded_files:

        uploaded_pdf = uploaded_files[0]

        # Store the temporary file path
        temporary_file_path = None

        try:

            # Save the uploaded PDF as a temporary local file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temporary_file:

                temporary_file.write(
                    uploaded_pdf.getvalue()
                )

                temporary_file_path = temporary_file.name


            # Call the existing backend RAG ingestion function
            with st.spinner(
                f"Processing {uploaded_pdf.name}..."
            ):

                ingest_rag_document(
                    temporary_file_path
                )


            # Display PDF processing confirmation
            st.toast(
                f"{uploaded_pdf.name} processed successfully.",
                icon="✅"
            )

        except Exception as error:

            # Display PDF processing error
            st.error(
                f"PDF processing failed: {error}"
            )

        finally:

            # Delete the temporary PDF after indexing
            if (
                temporary_file_path
                and os.path.exists(temporary_file_path)
            ):
                os.remove(temporary_file_path)


# Run this block after the user submits a text message
if user_input:

    # Save the user's message in Streamlit session state
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })


    # Display the user's message in the chat interface
    with st.chat_message("user"):
        st.text(user_input)


    # Pass the current thread ID to LangGraph
    # LangGraph uses this ID to save and retrieve conversation memory
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }


    # Assistant streaming block
    with st.chat_message("assistant"):

        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}


        def ai_only_stream():

            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages",
            ):

                # Lazily create & update the SAME status container
                # when any tool runs
                if isinstance(message_chunk, ToolMessage):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    )

                    if status_holder["box"] is None:

                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …",
                            expanded=True
                        )

                    else:

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )


                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content


        ai_message = st.write_stream(
            ai_only_stream()
        )


        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:

            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False
            )


    # Save the complete assistant response in Streamlit session state
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })