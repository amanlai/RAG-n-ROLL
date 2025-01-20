# standard library
import os
import time
from tempfile import NamedTemporaryFile
from typing import Iterator
import uuid
# third-party library
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from snowflake.snowpark import Session
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
# local
from .ingest import IngestData
from agent.graph import Agent
# from agent.test import TestAgent as Agent
load_dotenv()


CHAT_MODEL = os.getenv("CHAT_MODEL", "mistral-large-latest")  # "mistral-large2"
CHAT_MODEL_TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "e5-base-v2")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
VERBOSE = os.getenv("VERBOSE", "False") == "True"


def clear() -> None:
    st.session_state["chat_history"] = []


def create_answer(query: str) -> Iterator[str]:
    agent: CompiledStateGraph = st.session_state["agent"]
    chat_history: list[BaseMessage] = st.session_state["chat_history"]
    response = agent.invoke(
        st.session_state.to_dict(),
        {"configurable": {"thread_id": str(uuid.uuid4)}},
    )
    ai_response: AIMessage = response["messages"][-1]
    chat_history.extend((HumanMessage(content=query), ai_response))
    for w in ai_response.content.split():
        time.sleep(0.05)
        yield f"{w} "


def get_new_vector_store(uploaded_file: UploadedFile, ingester: IngestData) -> None:
    with st.spinner("Reading, splitting and embedding a file..."):
        with NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            vector_store = ingester.build_embeddings(tmp.name)
        os.remove(tmp.name)
        st.success("File chunked, embedded and indexed successfully.")
        st.session_state["vector_store"] = vector_store


def init_sidebar():
    start_session = st.button("Start/Restart Session")
    if start_session:
        # close old session
        if "session" in st.session_state:
            st.session_state["session"].close()
        # start new session
        connection_parameters = {
            "account": SNOWFLAKE_ACCOUNT,
            "user": SNOWFLAKE_USER,
            "password": SNOWFLAKE_PASSWORD,
        }
        session = Session.builder.configs(connection_parameters).create()
        st.session_state["session"] = session
        st.session_state["chat_history"] = []

    source = st.radio(
        "Select the data to use as context for your chatbot.",
        ["Use default", "Upload new"],
        index=None
    )
    if source:
        if source == "Use default":
            st.session_state["topic"] = "Snowflake Documentation"
            ingester = IngestData(
                session=st.session_state["session"],
                topic=st.session_state["topic"],
            )
            st.session_state["vector_store"] = ingester.get_vector_store()
        else:
            st.text_input("Topic", key="topic")
            uploaded_file = st.file_uploader("Upload a file", type="docx")
            add_data = st.button("Add Data")
            if add_data:
                ingester = IngestData(
                    session=st.session_state["session"],
                    topic=st.session_state["topic"],
                    model=EMBEDDING_MODEL
                )
                if uploaded_file:
                    get_new_vector_store(uploaded_file, ingester)
                else:
                    msg = "Must either upload a file or click the button to use existing data."
                    st.write(msg)
                    raise ValueError(msg)


def init_session_state() -> None:
    if "chat_history" not in st.session_state:
        clear()
    if "agent" not in st.session_state:
        agent = Agent(
            model=CHAT_MODEL,
            temperature=CHAT_MODEL_TEMPERATURE,
            topic=st.session_state["topic"],
            vector_store=st.session_state["vector_store"],
            verbose=VERBOSE
        )
        st.session_state["agent"] = agent.compile()


def init_main_page() -> None:
    if "topic" not in st.session_state:
        st.header("Your Chat Assistant about {}".format(st.session_state["topic"].title()))
        _, col = st.columns([3,1])
        with col:
            st.button("Reset chat history", key="reset_button", on_click=clear)
    init_session_state()
