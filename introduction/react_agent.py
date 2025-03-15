import datetime
import os

from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=500,
)

search_tool = TavilySearchResults(search_depth="basic")


@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format"""

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


tools = [search_tool, get_system_time]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
)


agent.invoke(
    "When was SpcaceX last launched? and how many days ago was that from this instant?"
)
