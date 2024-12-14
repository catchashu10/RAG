import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ragas import evaluation as eval
from ragas import EvaluationDataset
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    SemanticSimilarity,
    LLMContextPrecisionWithReference,
    ContextEntityRecall,
    ResponseRelevancy
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def evaluate_with_ragas(data, api_key, output_file_name, log=False):
    '''
    data = {
        "user_input": List[str],
        "response": List[str],
        "retrieved_contexts": List[List[str]],
        "reference": List[str]
    }
    '''

    df = pd.DataFrame(data)
    dataset = EvaluationDataset.from_pandas(df)
    os.environ["OPENAI_API_KEY"] = api_key

    # Initialize evaluator LLM and embeddings
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0.0))  # should temperature really be 0.0?
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    # Select metrics
    metrics = [
        LLMContextRecall(llm=evaluator_llm),
        FactualCorrectness(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        SemanticSimilarity(embeddings=evaluator_embeddings),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        ContextEntityRecall(),
        ResponseRelevancy(embeddings=evaluator_embeddings)
    ]

    # Run evaluation
    results = eval.evaluate(dataset=dataset, metrics=metrics)

    # Storing in a dataframe for quick analysis
    results = results.to_pandas()
    if log: print(results.head())

    results.to_csv(f"{output_file_name}.csv", index=False)

    # Average scores for each metric
    if log: print("Average Results:")
    mean_scores = results.select_dtypes(include=np.number).mean()
    if log: print(mean_scores)

    # List and calculate number of metrics
    metrics_names = mean_scores.index.tolist()
    values = mean_scores.values
    N = len(metrics_names)

    # Compute angles for radar
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values = np.append(values, values[0])

    # Radar Plot for the metrics
    fig, ax = plt.subplots(subplot_kw={'polar': True}, figsize=(8,6))
    plt.xticks(angles[:-1], metrics_names, color='grey', size=12)
    ax.set_ylim(0, 1)

    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.4)
    plt.title('RAGAS Metrics Radar Chart', size=16, y=1.1)

    plt.savefig(f"{output_file_name}.png", bbox_inches="tight")

    plt.show()

    return results

