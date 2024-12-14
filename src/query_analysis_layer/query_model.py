from pydantic import BaseModel, Field

class SubQuery(BaseModel):
    """You have performed query decomposition to break down the original query into distinct sub questions that you need to answer in order to answer the original question."""
    sub_queries: list[str] = Field(..., description="A list of sub queries that you need to answer in order to answer the original question.")

class ExpandedQuery(BaseModel):
    """You have performed query expansion to return multiple versions of the query with different phrasings."""
    expanded_queries: list[str] = Field(..., description="A list of expanded queries with different phrasings.")

class HyDE(BaseModel):
    """You have answered the user question as though you were writing a paragraph in an essay."""
    hyde_queries: str = Field(..., description="The answer to the user question as though you were writing a paragraph in an essay.")

class StepBackPrompting(BaseModel):
    """You have written a more generic question that needs to be answered in order to answer the original question."""
    step_back_prompting_queries: str = Field(..., description="A list of more generic questions that need to be answered in order to answer the original question.")