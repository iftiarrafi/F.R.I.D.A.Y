
import os
import uuid
import tempfile

import streamlit as st

from backend import chatbot, get_all_threads
from tools import ingest_rag_document, rag_tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="F.R.I.D.A.Y",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    #MainMenu, footer, [data-testid="stHeader"] {
        visibility: hidden;
    }

    .stApp {
        background: var(--background-color);
    }

    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 900px;
        padding: 2rem 2rem 7rem 2rem;
    }

    [data-testid="stSidebar"] {
        background: var(--secondary-background-color);
        border-right: 1px solid rgba(128, 128, 128, 0.15);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1.1rem 0.8rem;
    }

    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 42px;
        text-align: left;
        border: none;
        border-radius: 9px;
        background: transparent;
        padding: 0.55rem 0.8rem;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(128, 128, 128, 0.14);
    }

    .brand {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.6px;
        padding: 0.4rem 0 1.5rem 0;
    }

    .brand-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 31px;
        height: 31px;
        margin-right: 8px;
        border-radius: 50%;
        background: #10a37f;
        color: white;
        font-size: 17px;
    }

    .sidebar-section {
        color: var(--text-color);
        opacity: 0.65;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 1.3rem 0 0.6rem 0.6rem;
    }

    .welcome {
        padding-top: 15vh;
        text-align: center;
    }

    .welcome h1 {
        font-size: clamp(1.8rem, 4vw, 2.5rem);
        font-weight: 600;
        letter-spacing: -1px;
        margin-bottom: 0.6rem;
    }

    .welcome p {
        color: var(--text-color);
        opacity: 0.65;
        font-size: 1rem;
    }

    .welcome-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }

    .suggestion {
        padding: 14px 16px;
        border: 1px solid rgba(128, 128, 128, 0.23);
        border-radius: 14px;
        min-height: 75px;
        font-size: 0.88rem;
        color: var(--text-color);
        background: var(--secondary-background-color);
    }

    [data-testid="stChatMessage"] {
        border: none;
        padding: 1rem 0.25rem;
        background: transparent;
    }

    [data-testid="stChatMessageContent"] {
        line-height: 1.75;
        font-size: 0.96rem;
        overflow-wrap: anywhere;
    }

    [data-testid="stChatMessageContent"] pre {
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }

    [data-testid="stChatMessageContent"] code {
        font-size: 0.88em;
    }

    [data-testid="stChatInput"] {
        max-width: 850px;
        margin-left: auto;
        margin-right: auto;
    }

    [data-testid="stChatInput"] > div {
        border-radius: 24px;
        border: 1px solid rgba(128, 128, 128, 0.35);
        background: var(--secondary-background-color);
    }

    [data-testid="stChatInput"] textarea {
        padding-top: 14px;
        padding-bottom: 14px;
        font-size: 0.95rem;
    }

    .caption-text {
        opacity: 0.55;
        font-size: 0.76rem;
        text-align: center;
        margin-top: 1.1rem;
    }
    /* Left-align all sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left !important;
        justify-content: flex-start !important;
        padding-left: 12px !important;
    }

    /* Left-align button labels and their inner containers */
    [data-testid="stSidebar"] .stButton > button > div {
        width: 100%;
        justify-content: flex-start !important;
        text-align: left !important;
    }

    /* Left-align sidebar text */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        text-align: left;
    }

    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 1rem 6rem 1rem;
        }

        .welcome {
            padding-top: 8vh;
        }

        [data-testid="stChatMessage"] {
            padding: 0.6rem 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PDF CONFIGURATION
# =========================================================

DB_PATH = os.path.abspath("faiss_db")


def index_exists():
    """Return True if a valid-looking FAISS index exists."""
    return (
        os.path.isfile(os.path.join(DB_PATH, "index.faiss"))
        and os.path.isfile(os.path.join(DB_PATH, "index.pkl"))
    )


def is_pdf_related_question(question):
    """
    Basic detection for document questions.

    General messages such as 'Hi' or 'I am Rafi' will not
    trigger PDF retrieval.
    """
    question = question.lower().strip()

    pdf_keywords = [
        "pdf",
        "document",
        "uploaded file",
        "uploaded document",
        "uploaded pdf",
        "this file",
        "this document",
        "this paper",
        "the paper",
        "according to the file",
        "according to the document",
        "according to the pdf",
        "in the file",
        "in the document",
        "in the pdf",
        "from the file",
        "from the document",
        "from the pdf",
        "summarize",
        "summary of",
        "summarise",
        "contents of",
        "key points",
        "key findings",
        "main points",
        "which page",
        "what does the paper",
        "what does the document",
        "explain the first",
        "explain the second",
        "explain the third",
        "explain that point",
        "explain that section",
        "previous point",
        "second point",
        "first point",
        "third point",
        "next point",
        "that paragraph",
        "that section",
        "that chapter",
    ]

    return any(keyword in question for keyword in pdf_keywords)


# =========================================================
# THREAD MANAGEMENT
# =========================================================

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    if not state or not state.values:
        return []

    return state.values.get("messages", [])


def convert_messages(messages):
    converted = []

    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue

        content = message.content

        if isinstance(content, str):
            converted.append({
                "role": role,
                "content": content,
            })
        else:
            converted.append({
                "role": role,
                "content": str(content),
            })

    return converted


def reset_chat():
    new_thread_id = generate_thread_id()

    st.session_state["thread_id"] = new_thread_id
    st.session_state["message_history"] = []

    add_thread(new_thread_id)


# =========================================================
# SESSION STATE
# =========================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "active_pdf_name" not in st.session_state:
    st.session_state["active_pdf_name"] = None

if "pdf_context_enabled" not in st.session_state:
    st.session_state["pdf_context_enabled"] = False

add_thread(st.session_state["thread_id"])


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <span class="">✳</span> F.R.I.D.A.Y
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "＋  New chat",
        key="new_chat",
        use_container_width=True,
    ):
        reset_chat()
        st.rerun()

    st.markdown(
        '<div class="sidebar-section">YOUR CHATS</div>',
        unsafe_allow_html=True,
    )

    unique_threads = list(
        dict.fromkeys(st.session_state["chat_threads"])
    )

    st.session_state["chat_threads"] = unique_threads

    for thread_id in reversed(unique_threads):

        messages = load_conversation(thread_id)
        title = "New conversation"

        for message in messages:
            if isinstance(message, HumanMessage):
                content = message.content

                if isinstance(content, str) and content.strip():
                    title = " ".join(content.split())

                if len(title) > 32:
                    title = title[:29] + "..."

                break

        if thread_id == st.session_state["thread_id"]:
            title = "●  " + title

        if st.button(
            title,
            key=f"thread_{thread_id}",
            use_container_width=True,
        ):
            st.session_state["thread_id"] = thread_id

            st.session_state["message_history"] = (
                convert_messages(load_conversation(thread_id))
            )

            st.rerun()

    st.markdown("---")

    if st.session_state["active_pdf_name"]:
        st.caption(
            f"Last uploaded PDF: "
            f"{st.session_state['active_pdf_name']}"
        )

    if index_exists():
        st.caption("PDF index: Available")
    else:
        st.caption("PDF index: Not available")

    st.caption("F.R.I.D.A.Y · AI-powered document assistant")


# =========================================================
# WELCOME SCREEN / CHAT HISTORY
# =========================================================

if not st.session_state["message_history"]:

    st.markdown(
        """
        <div class="welcome">
            <div class="welcome-icon">✳</div>
            <h1>What can I help with?</h1>
            <p>Ask a question, explore an idea, or chat with your documents.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="suggestion">
                <b>📄 Understand a document</b><br>
                Summarize a PDF or explain its key concepts.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="suggestion">
                <b>💡 Explore an idea</b><br>
                Ask questions and get clear explanations.
            </div>
            """,
            unsafe_allow_html=True,
        )

else:

    for message in st.session_state["message_history"]:

        with st.chat_message(
            message["role"],
            avatar="🧑‍💻" if message["role"] == "user" else "✳",
        ):
            st.markdown(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================

submission = st.chat_input(
    "Ask F.R.I.D.A.Y...",
    accept_file=True,
    file_type=["pdf"],
)


# =========================================================
# HANDLE SUBMISSION AND PDF UPLOAD
# =========================================================

user_input = ""

if submission:

    # IMPORTANT: submission.text can be None when only a PDF
    # is uploaded. Always normalize it to a string.
    user_input = (submission.text or "").strip()

    # Some Streamlit versions may return an empty file list.
    uploaded_files = submission.files or []

    pdf_uploaded_successfully = False

    if uploaded_files:

        for uploaded_pdf in uploaded_files:

            temporary_file_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                ) as temporary_file:

                    temporary_file.write(uploaded_pdf.getvalue())
                    temporary_file_path = temporary_file.name

                with st.spinner(
                    f"Reading and indexing {uploaded_pdf.name}..."
                ):
                    ingest_rag_document(temporary_file_path)

                if not index_exists():
                    raise RuntimeError(
                        "PDF processing finished, but the FAISS "
                        "index files were not found."
                    )

                st.session_state["active_pdf_name"] = uploaded_pdf.name
                st.session_state["pdf_context_enabled"] = True

                pdf_uploaded_successfully = True

                st.success(
                    f"PDF uploaded and indexed: {uploaded_pdf.name}"
                )

            except Exception as error:
                st.error(
                    f"Could not process {uploaded_pdf.name}: {error}"
                )

            finally:
                if (
                    temporary_file_path
                    and os.path.exists(temporary_file_path)
                ):
                    os.remove(temporary_file_path)

    # Uploading a PDF without typing a message requests a summary.
    if uploaded_files and not user_input and pdf_uploaded_successfully:
        user_input = "Summarize the PDF I just uploaded."


# =========================================================
# PROCESS USER MESSAGE
# =========================================================

if isinstance(user_input, str) and user_input.strip():

    user_input = user_input.strip()

    # Display the original user question.
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"],
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"],
        },
        "run_name": "chat_trace",
    }

    # Always start with a valid string.
    message_content = user_input

    # -----------------------------------------------------
    # DECIDE WHETHER PDF RETRIEVAL IS NEEDED
    # -----------------------------------------------------

    should_search_pdf = (
        index_exists()
        and st.session_state["pdf_context_enabled"]
        and is_pdf_related_question(user_input)
    )

    if should_search_pdf:

        try:
            with st.spinner("Searching your PDF..."):

                pdf_context = rag_tool.invoke({
                    "query": user_input
                })

            if isinstance(pdf_context, str) and pdf_context.strip():

                message_content = f"""
The user is asking about an uploaded PDF.

User question:
{user_input}

Retrieved PDF context:
{pdf_context}

Instructions:
1. Answer PDF-related questions using the retrieved context.
2. Do not claim the PDF is missing when context is provided.
3. Do not invent facts unsupported by the context.
4. If the answer is absent from the retrieved context, say so.
5. Answer unrelated questions normally.
6. Do not reproduce the entire context unless requested.
"""

            else:
                message_content = (
                    f"{user_input}\n\n"
                    "No usable PDF context was retrieved. "
                    "Do not invent information from the document."
                )

        except Exception as error:
            st.warning(f"PDF retrieval failed: {error}")

            # Retrieval failure must not prevent normal chat.
            message_content = user_input

    # Final guard against invalid LangChain message content.
    if not isinstance(message_content, str) or not message_content.strip():
        message_content = user_input

    # -----------------------------------------------------
    # STREAM CHATBOT RESPONSE
    # -----------------------------------------------------

    with st.chat_message("assistant", avatar="✳"):

        status_holder = {"box": None}

        def ai_only_stream():

            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=message_content)
                    ]
                },
                config=CONFIG,
                stream_mode="messages",
            ):

                # Display tool activity, not tool output.
                if isinstance(message_chunk, ToolMessage):

                    tool_name = (
                        getattr(message_chunk, "name", None)
                        or "tool"
                    )

                    if status_holder["box"] is None:

                        status_holder["box"] = st.status(
                            f"Using {tool_name}...",
                            expanded=True,
                        )

                    else:

                        status_holder["box"].update(
                            label=f"Using {tool_name}...",
                            state="running",
                            expanded=True,
                        )

                # Stream assistant output.
                if isinstance(message_chunk, AIMessage):

                    content = message_chunk.content

                    if isinstance(content, str) and content:
                        yield content

                    elif isinstance(content, list):

                        for block in content:

                            if (
                                isinstance(block, dict)
                                and block.get("type") == "text"
                            ):
                                text_content = block.get("text", "")

                                if text_content:
                                    yield text_content

        try:

            ai_message = st.write_stream(ai_only_stream)

        except Exception as error:

            st.error(f"Something went wrong: {error}")
            ai_message = ""

        finally:

            if status_holder["box"] is not None:

                status_holder["box"].update(
                    label="Tool activity finished",
                    state="complete",
                    expanded=False,
                )

    # Save the assistant response for the UI.
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": (
            ai_message
            if isinstance(ai_message, str) and ai_message.strip()
            else "I couldn't generate a response. Please try again."
        ),
    })
