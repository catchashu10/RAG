from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from query_analysis_layer.graph import (
    invoke_query
)
import cohere
from flashrank import Ranker, RerankRequest
from .generation import generate_from_retrieved_documents
import pprint

def get_embedding_model(path="all-MiniLM-L6-v2"):
    model = SentenceTransformer(path)
    return model

def get_corpus_embeddings(corpus, model, batch_size=1, show_progress_bar=True, convert_to_tensor=True):
    embeddings = model.encode(corpus, batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_tensor=convert_to_tensor)
    return embeddings

def get_query_embedding(query, model, convert_to_tensor=True):
    query_embedding = model.encode(query, convert_to_tensor=convert_to_tensor)
    return query_embedding

def get_retrieved_docs(query_embedding, corpus_embeddings, top_k=10):
    retrieved_docs = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    return retrieved_docs

def get_retrieved_docs_with_scores(query, query_idx, model, corpus, embeddings, top_k):
    # Without query analysis layer
    if query_idx == -1:
        query_embedding = get_query_embedding(query, model)
        retrieved_docs = get_retrieved_docs(query_embedding, embeddings, top_k)[0]
        return retrieved_docs
    
    query_selection = query[query_idx]
    retrieved_docs = []
    # 0 -> DECOMPOSITION
    # 1 -> EXPANSION
    # 2 -> HYDE
    # 3 -> STEP BACK PROMPTING
    if query_idx == 0 or query_idx == 3:
        resulting_query = ""
        for i in range(len(query_selection)):
            resulting_query = " ".join([resulting_query, query_selection[i]])
            query_embedding = get_query_embedding(resulting_query, model)
            retrieved_docs = get_retrieved_docs(query_embedding, embeddings, top_k)[0]
            retrieved_doc = corpus[retrieved_docs[0]['corpus_id']]
            resulting_query = " ".join([resulting_query, retrieved_doc])
            
    else:
        for query in query_selection:
            query_embedding = get_query_embedding(query, model)
            retrieved_docs.extend(get_retrieved_docs(query_embedding, embeddings)[0])
            
    return retrieved_docs

def evaluate_model(questions, query_graph, query_idx, answers, corpus, corpus_embeddings, model, top_k=5, with_cohere=False):
    """
    Evaluate the accuracy of the embedding model by checking if the correct answer 
    is in the top-k retrieved results for each question.
    
    Parameters:
    - questions: List of questions to query.
    - answers: List of corresponding correct answers for the questions.
    - corpus: List of all answers in the corpus.
    - corpus_embeddings: Precomputed embeddings for the corpus.
    - model: The SentenceTransformer model.
    - top_k: Number of top results to consider for accuracy calculation.
    
    Returns:
    - accuracy: The overall accuracy of the model.
    """
    correct_count = 0
    total_questions = len(questions)
    # co = cohere.Client("qlZQilLRRahYfRjX3wzOhrpUkD4cA3yjcMG8MbiW")
    ranker = Ranker(model_name="rank-T5-flan", cache_dir="/mnt/project_vol/foundation_model/opt")
    print(questions)
    for i, question in enumerate(questions):
        if query_idx != -1:
            results = get_retrieved_docs_with_scores(invoke_query(query_graph, question), query_idx, model, corpus, corpus_embeddings, top_k)
        else:
            results = get_retrieved_docs_with_scores(question, query_idx, model, corpus, corpus_embeddings, top_k)
        if with_cohere:
            result_to_corpus_id_map = {}
            documents = []
            final_results = []
            for ii, result in enumerate(results):
                result_to_corpus_id_map[corpus[result['corpus_id']]] = [result['corpus_id'], result['score']]
                documents.append({"id": ii, "text": corpus[result['corpus_id']]})
            # results_post_rerank = co.rerank(query=question, documents=documents, top_n = 1, model = "rerank-multilingual-v2.0")
            rerankrequest = RerankRequest(query=question, passages=documents)
            results_post_rerank = ranker.rerank(rerankrequest)
            for result in results_post_rerank:
                final_results.append({'corpus_id' : result_to_corpus_id_map[documents[int(result["id"])]["text"]][0], 'score' :  result["score"]})
                
            results = final_results
        result = (results[0])
        
        # Check if the correct answer is in the top-k results
        correct_answer = answers[i]
        # retrieved_answers = [corpus[result['corpus_id']] for result in results]
        retrieved_answers = [corpus[result['corpus_id']]]

        if correct_answer in retrieved_answers:
            correct_count += 1
    # print(correct_answer)
    # print("y" * 30)
    # print("y" * 30)
    # print("retrieved_answers")
    # for ans in retrieved_answers:
    #     print(ans)
    #     print("x"*30)
    # print("retrieved_answers", retrieved_answers)
    # Calculate accuracy
    accuracy = correct_count / total_questions
    return accuracy

def retrieval_augmented_generation(questions, answers, corpus, corpus_embeddings, model, top_k=5):
    total_questions = len(questions)
    
    ragas_data = {
        "user_input": questions,
        "response": [],
        "retrieved_contexts": [],
        "reference": answers
    }

    for i, question in enumerate(questions):
        results = get_retrieved_docs_with_scores(question, -1, model, corpus, corpus_embeddings, top_k)
        ragas_data["retrieved_contexts"].append([corpus[result['corpus_id']] for result in results])
        ragas_data["response"].append(generate_from_retrieved_documents(question, [corpus[result['corpus_id']] for result in results]))
        
    return ragas_data