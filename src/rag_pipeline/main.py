from dotenv import load_dotenv

load_dotenv()

from dataset_layer.dataset_load import load_dataset_from_huggingface

from query_analysis_layer.graph import (
    build_graph,
    invoke_query,
    save_graph
)

from rag import (
    get_embedding_model,
    get_corpus_embeddings,
    get_query_embedding,
    get_retrieved_docs_with_scores,
    evaluate_model,
    fine_tune_embedding,
    retrieval_augmented_generation
)

from ragas_layer import (
    evaluate_with_ragas
)

QUERY_IDX = -1

def save_graph_to_file():
    graph = build_graph("gpt")
    save_graph(graph, "graph.png")

def respond_to_prompt():
    # DATASET LAYER
    train_ds, test_ds, corpus = load_dataset_from_huggingface()
    # EMBEDDING MODEL LAYER
    model = get_embedding_model()
    embeddings = get_corpus_embeddings(corpus, model)
    # QUESTION ANALYSIS LAYER
    user_query = input("Enter your query: ")
    graph = build_graph("gpt")
    final_query = invoke_query(graph, user_query)
    # RETRIEVAL LAYER
    retrieved_docs = get_retrieved_docs_with_scores(final_query, QUERY_IDX, model, corpus, embeddings)
        
    print(retrieved_docs)
    
def evaluate_without_rerank():
    train_ds, test_ds, corpus = load_dataset_from_huggingface()
    model = get_embedding_model()
    embeddings = get_corpus_embeddings(corpus, model)
    graph = build_graph("gpt")
    accuracy = evaluate_model(test_ds["question"][:7], graph, QUERY_IDX, test_ds["answer"][:7], corpus, embeddings, model, 1)
    print(f"Accuracy: {accuracy}")
    
def evaluate_with_rerank():
    train_ds, test_ds, corpus = load_dataset_from_huggingface()
    model = get_embedding_model()
    embeddings = get_corpus_embeddings(corpus, model)
    graph = build_graph("gpt")
    accuracy = evaluate_model(test_ds["question"][:7], graph, QUERY_IDX, test_ds["answer"][:7], corpus, embeddings, model, 5, True)
    print(f"Accuracy: {accuracy}")
    
def evaluate_ragas():
    train_ds, test_ds, corpus = load_dataset_from_huggingface()
    model = get_embedding_model()
    embeddings = get_corpus_embeddings(corpus, model)
    ragas_data = retrieval_augmented_generation(test_ds["question"][:7], test_ds["answer"][:7], corpus, embeddings, model, 5)
    print(ragas_data)
    result = evaluate_with_ragas(ragas_data, "temp.csv")
    print(result)
    
def run_fine_tune():
    fine_tune_embedding()