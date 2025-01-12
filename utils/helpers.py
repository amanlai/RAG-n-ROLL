# standard library
import os
import time
from typing import AsyncGenerator

# third-party library
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
import streamlit as st

# local
from agent.graph import Agent
# from agent.test import TestAgent as Agent


TOPIC = os.getenv("TOPIC", "Snowflake, Mistral and Streamlit")
vector_store = None        # will fill up later


def build_page() -> None:
    st.header(f"Your Chat Assistant about {TOPIC.title()}")
    _, col = st.columns([3,1])
    with col:
        st.button("Reset chat history", key="reset_button", on_click=clear)


def clear() -> None:
    st.session_state["chat_history"] = []


async def create_answer(query: str) -> AsyncGenerator[str, None]:
    agent: CompiledStateGraph = st.session_state["agent"]
    chat_history: list[BaseMessage] = st.session_state["chat_history"]
    response = await agent.ainvoke(
        st.session_state.to_dict(),
        {"configurable": {"thread_id": "1"}}
    )
    ai_response: AIMessage = response["messages"][-1]
    chat_history.extend((HumanMessage(content=query), ai_response))
    for w in ai_response.content.split():
        time.sleep(0.05)
        yield f"{w} "


def initialize_session_state() -> None:

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "agent" not in st.session_state:
        agent = Agent(topic=TOPIC, vector_store=vector_store, verbose=True)
        st.session_state["agent"] = agent.compile()
