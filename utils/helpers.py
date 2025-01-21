# standard library
import os
import time
from tempfile import NamedTemporaryFile
from typing import Iterator
import uuid
# third-party library
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from snowflake.snowpark import Session
import streamlit as st
# local
from .ingest import IngestData
from agent.graph import Agent
# from agent.test import TestAgent as Agent


CHAT_MODEL = st.secrets.get("CHAT_MODEL", "mistral-large-latest")  # "mistral-large2"
CHAT_MODEL_TEMPERATURE = float(st.secrets.get("TEMPERATURE", 0.1))
EMBEDDING_MODEL = st.secrets.get("EMBEDDING_MODEL", "e5-base-v2")
SNOWFLAKE_ACCOUNT = st.secrets.get("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER = st.secrets.get("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = st.secrets.get("SNOWFLAKE_PASSWORD")
VERBOSE = st.secrets.get("VERBOSE", "False") == "True"
K = int(st.secrets.get("K", 5))


############ callback functions ############

def clear() -> None:
    st.session_state["chat_history"] = []


def handle_ingestion() -> None:
    if st.session_state["source"] == "Use default":
        st.session_state["topic"] = "Using the Python connector in Snowflake"
        ingester = IngestData(
            session=st.session_state["session"],
            topic="usingthepythonconnectorinsnowflake",
        )
        st.session_state["vector_store"] = ingester.get_vector_store()
    else:
        st.text_input(
            r"Topic / Title: $\\\textsf{\scriptsize Enter a concise and "
            r"descriptive title for the context about to be uploaded.}$",
            key="topic"
        )
        uploaded_file = st.file_uploader("Upload a file", type="docx")
        add_data = st.button("Add Data")
        if add_data:
            if uploaded_file:
                ingester = IngestData(
                    session=st.session_state["session"],
                    topic="".join(st.session_state["topic"].split()).lower().replace("-", "_"),
                    model=EMBEDDING_MODEL
                )
                with st.spinner("Reading, splitting and embedding a file..."):
                    with NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(uploaded_file.read())
                        vector_store = ingester.build_embeddings(tmp.name)
                    os.remove(tmp.name)
                    st.success("File chunked, embedded and indexed successfully.")
                    st.session_state["vector_store"] = vector_store
            else:
                msg = "Must either upload a file."
                st.write(msg)
                raise ValueError(msg)


############ layout helper functions ############


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


def init_agent() -> None:
    if "agent" not in st.session_state:
        agent = Agent(
            model=CHAT_MODEL,
            temperature=CHAT_MODEL_TEMPERATURE,
            topic=st.session_state["topic"],
            vector_store=st.session_state["vector_store"],
            k=K,
            verbose=VERBOSE
        )
        st.session_state["agent"] = agent.compile()


def display_chat_history() -> None:
    icons = {"ai": "❄️", "human": "👤"}
    # display chat history
    for message in st.session_state["chat_history"]:
        with st.chat_message(message.type, avatar=icons[message.type]):
            st.markdown(message.content)
    # new user query
    if prompt := st.chat_input("How may I assist you today?"):
        with st.chat_message("human", avatar=icons["human"]):
            st.markdown(prompt)
            st.session_state["input"] = prompt
        with st.chat_message("ai", avatar=icons["ai"]):
            with st.spinner("Thinking..."):
                st.write_stream(create_answer(prompt))


############ streamlit page layout functions ############


def init_sidebar():
    st.markdown(
        "<p style='font-size:12px;'>"
        "Please start a new session to use the chatbot. You can restart the session at any time.<br>"
        "Note that by start a new session, you will lose access to any previous chat history and uploaded context."
        "</p>",
        unsafe_allow_html=True
    )
    start_session = st.button("Start Session")
    if start_session:
        # close old session
        if "session" in st.session_state:
            st.session_state["session"].close()
        if "agent" in st.session_state:
            del st.session_state["agent"]
        if "vector_store" in st.session_state:
            del st.session_state["vector_store"]
        # start new session
        connection_parameters = {
            "account": SNOWFLAKE_ACCOUNT,
            "user": SNOWFLAKE_USER,
            "password": SNOWFLAKE_PASSWORD,
            "paramstyle": "pyformat"
        }
        session = Session.builder.configs(connection_parameters).create()
        st.session_state["session"] = session
        clear()

        st.radio(
            "Select the data to use as context for your chatbot.",
            ["Use default", "Upload new"],
            index=None,
            key="source",
            on_change=handle_ingestion
        )


def init_main_page() -> None:
    if "chat_history" not in st.session_state:
        clear()
    if "topic" in st.session_state and st.session_state["topic"] != "":
        st.header("Your Chat Assistant about {}".format(st.session_state["topic"].title()))
        _, col = st.columns([3,1])
        with col:
            st.button("Reset chat history", key="reset_button", on_click=clear)
        if "vector_store" in st.session_state:
            init_agent()
        display_chat_history()
