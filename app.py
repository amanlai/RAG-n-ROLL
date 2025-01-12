# third-party library
import streamlit as st

# local
from utils.helpers import build_page, create_answer, initialize_session_state

st.set_page_config(
    page_title="RAG 'n' ROLL Amp up Search",
    page_icon="./public/cottontail.png"
)


def main():

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
            st.write_stream(create_answer(prompt))


if __name__ == '__main__':
    build_page()
    initialize_session_state()
    main()
