# standard library
from typing import Annotated, Literal, TypedDict
# third-party library
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


class State(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    messages: Annotated[list[BaseMessage], add_messages]


class Config(TypedDict):
    thread_id: str


class Agent:
    def __init__(
        self,
        model: str,
        temperature: float,
        topic: str,
        vector_store: VectorStore,
        k: int = 5,
        verbose: bool = True,
    ) -> None:
        self.verbose = verbose
        self.llm: BaseChatModel = ChatMistralAI(
            model=model,
            temperature=temperature,
            max_retries=2,
        )
        self.tools = get_tools(topic, vector_store, k=k)
        self.tool_node = ToolNode(self.tools)
        llm_with_tools = self.llm.bind_tools(self.tools)
        self.answer_chain = self.get_chain(llm_with_tools)
        self.workflow = self.get_graph()

    def get_chain(self, llm: Runnable) -> Runnable:
        system_prompt = (
            "You are a helpful AI chat assistant with RAG capabilities. "
            "When a user asks you a question, you will use 'search-for-context' tool to get text relevant to the question. "
            "Use that context with the user's chat history to provide a summary that addresses the user's question. "
            "Ensure the answer is coherent, concise, and directly relevant to the user's question.\n"
            "If the user asks a generic question which cannot be answered with the given context or chat_history,"
            "just say 'I don\'t know the answer to that question.'\n"
            "Don't say things like 'according to the provided context'."
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
