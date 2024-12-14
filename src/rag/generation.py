from langchain_openai import ChatOpenAI

def generate_from_retrieved_documents(query, retrieved_documents):
    prompt = """
        Generate a response to this query: {query} using the following retrieved documents:
        {retrieved_documents}
    """
    
    prompt = prompt.format(query=query, retrieved_documents="\n".join(retrieved_documents))
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke(prompt)
    
    return response.content