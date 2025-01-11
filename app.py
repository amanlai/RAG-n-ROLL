# standard library
import os

# third-party library
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
import streamlit as st

# local
from agent.graph import Agent

TOPIC = os.getenv("TOPIC")
vector_store = None        # will fill up later


st.set_page_config(
    page_title="RAG 'n' ROLL Amp up Search",
    page_icon="./public/cottontail.png"
)


def create_answer(query: str) -> str:
    agent: CompiledStateGraph = st.session_state["agent"]
    chat_history: list[BaseMessage] = st.session_state["chat_history"]
    response = agent.invoke(
        dict(st.session_state),
        {"configurable": {"thread_id": "1"}}
    )
    ai_response: AIMessage = response["messages"][-1]
    chat_history.extend((HumanMessage(content=query), ai_response))
    return ai_response.content


def main():
    st.header(f"Your Chat Assistant about {TOPIC.title()}")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "agent" not in st.session_state:
        agent = Agent(topic=TOPIC, vector_store=vector_store)
        st.session_state["agent"] = agent.compile()

    # display chat history
    for message in st.session_state["chat_history"]:
        with st.chat_message(message.type):
            st.markdown(message.content)

    # new user query
    if prompt := st.chat_input("How may I assist you today?"):
        with st.chat_message("human"):
            st.markdown(prompt)
            st.session_state["input"] = prompt
        with st.chat_message("ai"):
            answer = create_answer(prompt)
            st.markdown(answer)


if __name__ == '__main__':
    main()
