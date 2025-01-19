# third-party library
import streamlit as st
# local
from utils.helpers import (
    build_page,
    create_answer,
    initialize_session_state,
    initialize_sidebar
)

st.set_page_config(
    page_title="RAG 'n' ROLL Amp up Search",
    page_icon="./public/cottontail.png"
)


def main():

    with st.sidebar:
        initialize_sidebar()

    build_page()
    initialize_session_state()

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


if __name__ == '__main__':
    main()
