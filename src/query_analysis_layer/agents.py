from .state_definition import *
from .prompts import *
from .query_model import *
from langchain_openai import ChatOpenAI

def query_decomposition_agent(state: ModelInput, llm: ChatOpenAI):
    prompt = decomposition_system.format(query=state["query"])
    response = llm.with_structured_output(SubQuery).invoke(prompt)
    return {
        "decomposed_queries": response.sub_queries
    }

def query_expansion_agent(state: ModelInput, llm: ChatOpenAI):
    prompt = expansion_system.format(query=state["query"])
    response = llm.with_structured_output(ExpandedQuery).invoke(prompt)
    return {
        "expanded_queries": response.expanded_queries
    }

def hyde_agent(state: ModelInput, llm: ChatOpenAI):
    prompt = hyde_system.format(query=state["query"])
    response = llm.with_structured_output(HyDE).invoke(prompt)
    hyde_queries = " ".join(state["query"] for _ in range(3)) + response.hyde_queries
    return {
        "hyde_queries": hyde_queries
    }

def step_back_prompting_agent(state: ModelInput, llm: ChatOpenAI):
    prompt = step_back_prompting_system.format(query=state["query"])
    response = llm.with_structured_output(StepBackPrompting).invoke(prompt)
    return {
        "step_back_prompting_queries": [response.step_back_prompting_queries, state["query"]]
    }
    
def combine_queries(state: CombinedModelInput):
    decomposed_queries = state["decomposed_queries"] if state["decomposed_queries"] else []
    expanded_queries = state["expanded_queries"] if state["expanded_queries"] else []
    hyde_queries = [state["hyde_queries"]] if state["hyde_queries"] else []
    step_back_prompting_queries = state["step_back_prompting_queries"] if state["step_back_prompting_queries"] else []
    return {
        "messages": [decomposed_queries, expanded_queries, hyde_queries, step_back_prompting_queries]
    }