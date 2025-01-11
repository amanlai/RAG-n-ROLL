# standard library
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

# third-party library
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool, create_retriever_tool
from langchain_core.vectorstores import VectorStore

TIMEZONE = ZoneInfo("US/Pacific")


@tool("get-today-tool")
def get_today() -> tuple[
        Literal["Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"],
        datetime]:
    """
    Useful for getting today's weekday name and today's date as a Python \
    datetime object. No input required. Output is a tuple of the weekday \
    name and the datetime.
    """
    today = datetime.now(TIMEZONE)
    weekday = today.strftime("%A")
    return weekday, today


def get_tools(topic: str, vector_store: VectorStore, k: int = 5) -> BaseRetriever:
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    description = (
        f"Search for information about {topic}. "
        f"For any questions about {topic}, you must use this tool. "
        "Use the noun or the phrase most similar to the search."
    )
    retriever_tool = create_retriever_tool(
        retriever,
        name="search-vector-store",
        description=description,
    )
    return [get_today, retriever_tool]
