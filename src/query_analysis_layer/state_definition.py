from typing_extensions import TypedDict

class State(TypedDict):
    query: str
    decomposed_queries: list[str]
    expanded_queries: list[str]
    hyde_queries: str
    step_back_prompting_queries: list[str]
    messages: list[str]

class ModelInput(TypedDict):
    query: str

class DecomposedModelOutput(TypedDict):
    decomposed_model: list[str]

class ExpandedModelOutput(TypedDict):
    expanded_model: list[str]

class HyDEModelOutput(TypedDict):
    hyde_model: str

class StepBackPromptingModelOutput(TypedDict):
    step_back_prompting_model: str

class CombinedModelInput(TypedDict):
    decomposed_queries: list[str]
    expanded_queries: list[str]
    hyde_queries: str
    step_back_prompting_queries: list[str]

class CombinedModelOutput(TypedDict):
    messages: list[str]