import os
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode
from langchain_community.document_loaders import WebBaseLoader

from src.llm import llm_gpt_4o, llm_gpt_4o_mini
from src.utils import PageSessionState, BaseChatAgent

# Load environment variables
load_dotenv()

def get_website(url: str) -> str:
    loader = WebBaseLoader(url)
    loader.requests_kwargs = {'verify':False}
    docs = loader.load()
    # docs[0].__dict__['page_content']
    return docs


class BereanAgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]



def get_vision_content() -> str:
    """Get the content of the Berean Vision page"""
    url = "https://bereancc.com/our-vision"
    docs = get_website(url)
    return docs[0].__dict__['page_content'].strip()

def get_resources_content() -> str:
    """Get the content of the Berean Resources page which includes recommended books"""
    url = "https://bereancc.com/resources"
    docs = get_website(url)
    return docs[0].__dict__['page_content'].strip()

tools =[
    get_vision_content,
    get_resources_content,
]


def create_berean_agent_graph():
    current_date = datetime.now().strftime("%B %d, %Y")
    system_prompt = f"""You are a helpful assistant for Berean Community Church. Today's date is {current_date}

    You have access to tools to fetch the latest events and resources from the church website.
    """


    def call_model(state: BereanAgentState):
        llm = llm_gpt_4o_mini.bind_tools(tools)
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: BereanAgentState) -> Literal["tools", "__end__"]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError(f"Expected AIMessage, got {type(last_message)}")
        if last_message.tool_calls:
            return "tools"
        return "__end__"

    workflow = StateGraph(BereanAgentState)

    # Add nodes
    workflow.add_node("model", call_model)
    workflow.add_node("tools", ToolNode(tools))

    # Add edges
    workflow.add_edge(START, "model")
    workflow.add_edge("tools", "model")
    workflow.add_conditional_edges("model", should_continue)

    return workflow.compile()


def initialize_page_state(page_state: PageSessionState):
    """Initialize all required state variables for the page"""
    if "messages" not in page_state:
        page_state["messages"] = []
    if "graph" not in page_state:
        page_state["graph"] = create_berean_agent_graph()


class BereanChatAgent(BaseChatAgent):
    def __init__(self):
        super().__init__("ukg", create_berean_agent_graph)


def main():
    st.title("Berean Assistant")
    st.markdown("""
    I can help you with:
    - Answering questions about Berean's Vision and Recommended books
    """)

    agent = BereanChatAgent()
    agent.run()


if __name__ == "__main__":
    main()
