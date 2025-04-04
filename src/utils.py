import os
import json
import base64
import tempfile
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.schema import HumanMessage
import logging
from src.st_callable_util import get_streamlit_cb

project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"

dotenv_path = find_dotenv()
# load up the entries as environment variables
load_dotenv(dotenv_path)

logging.basicConfig(
    filename="utils.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def stream_chatgpt_output(response):
    """
    This function is used to stream the output of the chatgpt model.
    It is called every time a new chunk is received from the API.

    Returns: list of output tokens
    """
    res_box = st.empty()
    output = []
    for chunk in response:
        try:
            # build up list of output tokens
            output.append(chunk["choices"][0]["delta"]["content"])
            # write output every time we receive a chunk to mimic streaming
            res_box.markdown("".join(output))
        except Exception:
            # some chunks don't have a 'choices[0]['delta']['content'] key, so we just
            # ignore them
            pass
    return output


def wrap_text(df, column_name):
    """
    This function is used to wrap the text in a dataframe using custom CSS
    and alternates each row color as gray.
    """

    # def alternate_row_color(row):
    #     color = "background-color: lightgray" if row.name % 2 else ""
    #     return [color] * len(row)

    # def highlight_true(cell):
    #     color = "color: green; font-weight: bold" if cell else ""
    #     return color

    # return (
    #     df.style.set_properties(
    #         **{"white-space": "pre-wrap", "word-break": "break-word"}
    #     )
    #     .apply(alternate_row_color, axis=1)
    #     .applymap(
    #         highlight_true,
    #         subset=pd.IndexSlice[:, df.select_dtypes(include=[bool]).columns],
    #     )
    # )
    def highlight_row(row):
        color = (
            "background-color: lightgreen; font-weight: bold"
            if row["answer"]
            else ""
        )
        return [color] * len(row)

    return df.style.set_properties(
        **{"white-space": "pre-wrap", "word-break": "break-word"}
    ).apply(highlight_row, axis=1)



def get_page_state_key(base_key: str, page_name: str) -> str:
    """
    Generate a page-specific session state key.

    Args:
        base_key: The original key name
        page_name: The name of the current page

    Returns:
        A unique key specific to the current page
    """
    return f"{page_name}__{base_key}"


class PageSessionState:
    """
    Manages page-specific session state variables.
    """

    def __init__(self, page_name: str):
        self.page_name = page_name

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the page's session state."""
        full_key = get_page_state_key(key, self.page_name)
        return full_key in st.session_state

    def __getitem__(self, key: str):
        """Get a value from the page's session state."""
        full_key = get_page_state_key(key, self.page_name)
        return st.session_state[full_key]

    def __setitem__(self, key: str, value):
        """Set a value in the page's session state."""
        full_key = get_page_state_key(key, self.page_name)
        st.session_state[full_key] = value


class BaseChatAgent:
    """Base class for all chat-based agents"""

    def __init__(self, page_name: str, create_graph_func: Callable):
        """
        Initialize the base chat agent.

        Args:
            page_name: Name of the page/agent (e.g., "ukg", "math", "recruiting")
            create_graph_func: Function to create the specific graph for this agent
        """
        self.page_name = page_name
        self.create_graph_func = create_graph_func
        self.page_state = PageSessionState(page_name)

    def initialize_state(
        self, additional_state: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize all required state variables.

        Args:
            additional_state: Optional dictionary of additional state variables to initialize
        """
        if "messages" not in self.page_state:
            self.page_state["messages"] = []
        if "graph" not in self.page_state:
            self.page_state["graph"] = self.create_graph_func()

        # Initialize any additional state variables
        if additional_state:
            for key, default_value in additional_state.items():
                if key not in self.page_state:
                    self.page_state[key] = default_value

    def prepare_invoke_args(self, messages: list) -> Dict[str, Any]:
        """
        Prepare arguments for graph.invoke(). Override this in subclasses if needed.

        Args:
            messages: List of chat messages

        Returns:
            Dictionary of arguments for graph.invoke()
        """
        return {"messages": messages}

    def display_chat_history(self):
        """Display all previous messages in the chat."""
        messages = self.page_state["messages"]
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            st.chat_message(role).write(msg.content)

    def handle_chat(self):
        """Handle the chat interaction."""
        messages = self.page_state["messages"]
        graph = self.page_state["graph"]

        # Create a container for new messages
        response_container = st.container()

        # Handle new messages
        if prompt := st.chat_input():
            with response_container:
                # Add new message to the state
                messages.append(HumanMessage(content=prompt))
                self.page_state["messages"] = messages

                st.chat_message("user").write(prompt)

                with st.chat_message("assistant"):
                    msg_placeholder = st.empty()
                    st_callback = get_streamlit_cb(st.empty())

                    try:
                        invoke_args = self.prepare_invoke_args(messages)
                        response = graph.invoke(
                            invoke_args,
                            config={"callbacks": [st_callback]},
                        )

                        last_msg = response["messages"][-1].content
                        messages.append(response["messages"][-1])
                        self.page_state["messages"] = messages
                        msg_placeholder.write(last_msg)

                    except Exception as e:
                        error_msg = f"An error occurred: {str(e)}"
                        st.error(error_msg)
                        msg_placeholder.write(
                            "I encountered an error processing your request."
                        )

    def run(self):
        """Main method to run the chat agent."""
        self.initialize_state()
        self.display_chat_history()
        self.handle_chat()
