# third-party library
import streamlit as st
# local
from utils.helpers import init_main_page, init_sidebar

st.set_page_config(
    page_title="RAG 'n' ROLL Amp up Search",
    page_icon="./public/cottontail.png"
)


if __name__ == "__main__":
    with st.sidebar:
        init_sidebar()
    init_main_page()
