from typing import List, Sequence

from chains import generation_chain, reflection_chain
from dotenv import load_dotenv
from langchain.graph import END, MessageGraph
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

load_dotenv()

graph = MessageGraph()

REFLECT = "reflect"
GENERATE = "generate"


def generate_node(state):
    return generation_chain.invoke({"messages": state})


def reflect_node(state):
    response = reflection_chain.invoke({"messages": state})
    return [HumanMessage(content=response.content)]


graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)
