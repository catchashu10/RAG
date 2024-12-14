decomposition_system = """Perform query decomposition. Given a user question break it down into distinct sub questions that you need to answer in order to answer the original question. User question: {query}
"""

expansion_system = """Perform query expansion. If there are multiple ways of phrasing a user question or common synonyms for the key words in the question, make sure to return multiple versions of the query with different phrasings. Always encapsulate the entire original query in the expanded queries. If there are words you are not familiar with, do not try to rephrase those. Return atleast 3 different versions of the query. User question: {query}"""

hyde_system = """You are an expert about a wide range of topics. Answer as though you were writing a paragraph in an essay. User question: {query}"""

step_back_prompting_system = """Given a user question, write a more generic question that needs to be answered in order to answer the original question. In effect, think of it as zooming out from the question posed by the user. User question: {query}"""