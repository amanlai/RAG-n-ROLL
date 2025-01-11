# standard library
from typing import Annotated, Literal, TypedDict

# third-party library
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.vectorstores import VectorStore
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Checkpointer

# local
from .tools import get_tools

load_dotenv()


MODEL_NAME = "mistral-large-latest"  # "mistral-large2"
TEMPERATURE = 0.1


class State(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    messages: Annotated[list[BaseMessage], add_messages]


class Config(TypedDict):
    thread_id: str


class Agent:
    def __init__(
        self,
        topic: str,
        vector_store: VectorStore,
        verbose: bool = True,
    ) -> None:
        self.verbose = verbose
        self.llm: BaseChatModel = ChatMistralAI(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_retries=2,
        )
        self.tools = get_tools(topic, vector_store)
        self.tool_node = ToolNode(self.tools)
        llm_with_tools = self.llm.bind_tools(self.tools)
        self.answer_chain = self.get_chain(llm_with_tools)
        self.workflow = self.get_graph()

    def get_chain(self, llm: Runnable) -> Runnable:
        system_prompt = (
            "You are a helpful assistant. "
            "Be concise and accurate."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt),
                MessagesPlaceholder("chat_history", optional=True),
                HumanMessagePromptTemplate.from_template("{input}"),
                MessagesPlaceholder("messages", optional=True),
            ]
        )
        chain = prompt | llm
        return chain

    def should_continue(self, state: State) -> Literal["continue", "exit"]:
        last_message: AIMessage = state["messages"][-1]
        if last_message.tool_calls:
            return "continue"
        else:
            return "exit"

    def run_llm(self, state: State, config: RunnableConfig) -> dict:
        response: AIMessage = self.answer_chain.invoke(state, config)
        if self.verbose:
            response.pretty_print()
        return {"messages": [response]}

    def run_tools(self, state: State, config: RunnableConfig) -> dict:
        response = self.tool_node.invoke(state, config)
        if self.verbose:
            for m in response["messages"]:
                m.pretty_print()
        return response

    def get_graph(self) -> StateGraph:
        workflow = StateGraph(State, config_schema=Config)
        workflow.add_node("agent", self.run_llm)
        workflow.add_node("tools", self.run_tools)
        workflow.add_edge("__start__", "agent")
        workflow.add_conditional_edges(
            "agent",
            path=self.should_continue,
            path_map={"continue": "tools", "exit": "__end__"}
        )
        workflow.add_edge("tools", "agent")
        return workflow

    def compile(self, checkpointer: Checkpointer | None = None) -> CompiledStateGraph:
        return self.workflow.compile(checkpointer)
