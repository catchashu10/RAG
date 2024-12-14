from langgraph.graph import StateGraph, START, END
from functools import partial
from typing import Literal
from langchain_openai import ChatOpenAI
import pprint

from .agents import *

def build_graph(embedding_model: Literal["gpt", "sentence_transformer", "claude"]):
    graph_builder = StateGraph(State)

    # Currently only GPT-4o is supported
    decomp_llm = ChatOpenAI(model="gpt-4o")
    expan_llm = ChatOpenAI(model="gpt-4o")
    hyde_llm = ChatOpenAI(model="gpt-4o")
    step_back_llm = ChatOpenAI(model="gpt-4o")

    graph_builder.add_node("query_decomposition_agent", partial(query_decomposition_agent, llm=decomp_llm))
    graph_builder.add_node("query_expansion_agent", partial(query_expansion_agent, llm=expan_llm))
    if embedding_model != "sentence_transformer":
        graph_builder.add_node("hyde_agent", partial(hyde_agent, llm=hyde_llm))
    graph_builder.add_node("step_back_prompting_agent", partial(step_back_prompting_agent, llm=step_back_llm))

    graph_builder.add_node("combine_queries_agent", combine_queries)

    graph_builder.add_edge(START, "query_decomposition_agent")
    graph_builder.add_edge(START, "query_expansion_agent")
    if embedding_model != "sentence_transformer":
        graph_builder.add_edge(START, "hyde_agent")
    graph_builder.add_edge(START, "step_back_prompting_agent")

    graph_builder.add_edge("query_decomposition_agent", "combine_queries_agent")
    graph_builder.add_edge("query_expansion_agent", "combine_queries_agent")
    if embedding_model != "sentence_transformer":
        graph_builder.add_edge("hyde_agent", "combine_queries_agent")
    graph_builder.add_edge("step_back_prompting_agent", "combine_queries_agent")

    graph_builder.add_edge("combine_queries_agent", END)

    graph = graph_builder.compile()
    return graph

def invoke_query(graph, query):
    final_state = graph.invoke({"query": query})
    pprint.pp(final_state)
    return final_state["messages"]

def save_graph(graph, output_path):
    png_data = graph.get_graph().draw_mermaid_png()
    
    with open(output_path, 'wb') as file:
        file.write(png_data)