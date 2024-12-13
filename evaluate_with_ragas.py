import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from datasets import load_dataset
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

# Mock data for testing
mock_data = {
    "user_input": ["Where is the Eiffel Tower located?", "Who was Albert Einstein?"],
    "response": [
        "The Eiffel Tower is located in Paris.",
        "Albert Einstein was a German-born theoretical physicist."
    ],
    "retrieved_contexts": [
        ["The Eiffel Tower, an iron lattice tower in Paris, France."],
        ["Albert Einstein (1879–1955) was a German-born theoretical physicist."]
    ],
    "reference": [
        "The Eiffel Tower is located in Paris, France.",
        "Albert Einstein was a German-born theoretical physicist known for the theory of relativity."
    ]
}

df = pd.DataFrame(mock_data)
eval_dataset = EvaluationDataset.from_pandas(df)

# Load evaluation data
# TODO: Replace with our dataset
# dataset = load_dataset(
#     "explodinggradients/amnesty_qa",
#     "english_v3",
#     trust_remote_code=True
# )
# eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])

# OpenAI key to be set in the environment
os.environ["OPENAI_API_KEY"] = "<REPLACE ME>"

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
results = eval.evaluate(dataset=eval_dataset, metrics=metrics)

# Storing in a dataframe for quick analysis
results = results.to_pandas()
print(results.head())

results.to_csv("evaluation_results.csv", index=False)

# Average scores for each metric
print("Average Results:")
mean_scores = results.select_dtypes(include=np.number).mean()
print(mean_scores)

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

plt.savefig("radar_chart.png", bbox_inches="tight")

plt.show()
