import os
from datetime import datetime
from typing import TypedDict, Annotated, List, Literal, Dict, Any, Optional, Callable

from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from langchain_community.document_loaders import WebBaseLoader

from src.llm import llm_gpt_4o, llm_gpt_4o_mini
from src.utils import PageSessionState, BaseChatAgent, check_password

# Load environment variables
load_dotenv()

if not check_password():
    st.stop()


def get_website(url: str) -> str:
    """
    Fetch and extract content from a given URL.
    
    Args:
        url: The URL to fetch content from
        
    Returns:
        The extracted text content from the webpage
        
    Raises:
        Exception: If there's an error fetching or parsing the webpage
    """
    try:
        loader = WebBaseLoader(url)
        loader.requests_kwargs = {'verify': False}
        docs = loader.load()
        return docs[0].__dict__['page_content'].strip()
    except Exception as e:
        st.error(f"Error fetching content from {url}: {str(e)}")
        return f"Unable to fetch content: {str(e)}"


class BereanAgentState(TypedDict):
    """Type definition for the Berean Agent state."""
    messages: Annotated[List[AnyMessage], add_messages]


class BereanWebContent:
    """Helper class for fetching Berean Church web content."""
    
    BASE_URL = "https://bereancc.com"
    
    @staticmethod
    def get_page_content(path: str, description: str) -> Callable[[], str]:
        """
        Create a function that fetches content from a specific Berean webpage.
        
        Args:
            path: The URL path after the base URL
            description: Description of the content being fetched
            
        Returns:
            A function that when called, fetches the content
        """
        def fetch_content() -> str:
            """
            Get the content of the {description}.
            
            Returns:
                The webpage content as text
            """
            url = f"{BereanWebContent.BASE_URL}/{path}"
            return get_website(url)
            
        # Set the docstring and function name properly
        fetch_content.__doc__ = f"Get the content of the {description}."
        fetch_content.__name__ = f"get_{path.replace('-', '_')}_content"
        
        return fetch_content


# Define tool functions using the helper class
get_resources_content = BereanWebContent.get_page_content("resources", "Berean Resources page which includes recommended books")
get_what_we_believe_content = BereanWebContent.get_page_content("what-we-believe", "Berean What We Believe page")
get_distinctives_content = BereanWebContent.get_page_content("distinctives", "Berean Distinctives page")
get_vision_content = BereanWebContent.get_page_content("our-vision", "Berean Vision page")
get_membership_covenant_content = BereanWebContent.get_page_content("membership-covenant", "Berean Membership Covenant page")
get_meeting_times_content = BereanWebContent.get_page_content("meeting-times-amp-location", "Berean Meeting Times page")
get_what_is_the_gospel_content = BereanWebContent.get_page_content("what-is-the-gospel", "Berean What is the Gospel page")

# List of available tools for the agent
tools = [
    get_resources_content,
    get_what_we_believe_content,
    get_distinctives_content,
    get_vision_content,
    get_membership_covenant_content,
    get_meeting_times_content,
    get_what_is_the_gospel_content,
]


def create_berean_agent_graph() -> Any:
    """
    Create and return a compiled state graph for the Berean agent.
    
    Returns:
        A compiled state graph that can be executed
    """
    current_date = datetime.now().strftime("%B %d, %Y")
    system_prompt = f"""You are a helpful assistant for Berean Community Church. Today's date is {current_date}

    You have access to tools to fetch the latest content from the Berean Community Church website.
    """

    def call_model(state: BereanAgentState) -> Dict[str, List[AIMessage]]:
        """
        Call the LLM with the current conversation state.
        
        Args:
            state: The current conversation state
            
        Returns:
            Updated state with the model's response
        """
        llm = llm_gpt_4o_mini.bind_tools(tools)
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: BereanAgentState) -> Literal["tools", "__end__"]:
        """
        Determine if the conversation should continue or end.
        
        Args:
            state: The current conversation state
            
        Returns:
            Next node to route to ("tools" or "__end__")
            
        Raises:
            TypeError: If the last message is not an AIMessage
        """
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


def initialize_page_state(page_state: PageSessionState) -> None:
    """
    Initialize all required state variables for the page.
    
    Args:
        page_state: The current session state object
    """
    if "messages" not in page_state:
        page_state["messages"] = []
    if "graph" not in page_state:
        page_state["graph"] = create_berean_agent_graph()


class BereanChatAgent(BaseChatAgent):
    """Chat agent implementation for the Berean Church assistant."""
    
    def __init__(self) -> None:
        """Initialize the Berean chat agent."""
        super().__init__("ukg", create_berean_agent_graph)


def main() -> None:
    """Main function to run the Streamlit application."""
    st.title("Berean Assistant")
    st.markdown("""
    I can help you with:
    - Answering questions about Berean's Vision and Recommended books
    - Information about church beliefs, distinctives, and meeting times
    - Details about membership and the gospel
    """)

    agent = BereanChatAgent()
    agent.run()


if __name__ == "__main__":
    main()
