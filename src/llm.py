import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
# load up the entries as environment variables
load_dotenv(dotenv_path)

llm_gpt_4o = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0,
    max_tokens=8192,
)

llm_gpt_4o_mini = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    max_tokens=8192,
)
