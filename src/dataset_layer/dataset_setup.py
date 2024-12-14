import json
import random
from typing import Dict, List, Tuple
from datasets import Dataset, DatasetDict


"""
    This code takes the formatted JSON input data and process it into the final 'train', 'val' and 'test' data in the format that will be 
    Input Data:
        Input Dataset should be of the following format:
            {
                "queries": {
                    "query_uuid": "query string",
                    "query_uuid": "query string",
                    ....
                },
                "corpus": {
                    "corpus_uuid": "corpus_string",
                    "corpus_uuid": "corpus_strig",
                    ...
                },
                "relevant_docs": {
                    "query_uuid": ["corpus_uuid"],
                    "query_uuid": ["corpus_uuid"],
                    ...
                }
            }

    create_dataset.py takes the given data consisting of all the queries, corpus and relevant_docs and creates the final dataset needed
    in the following format:
        DatasetDict({
            train: Dataset({
                features: ['anchor', 'positive'],
                num_rows: 24145
            })
            val: DatasetDict({
                features: ['anchor', 'positive'],
                num_rows: 3545
            })
            evaluator_val: Dict({
                queries: {"query_uuid": "query string", .....}
                corpus: {"corpus_uuid": "corpus string", ....}
                relevant_docs: {"query_uuid": ["corpus_uuid"], ....}
            })
            evaluator_test: Dict({
                queries: {"query_uuid": "query string", .....}
                corpus: {"corpus_uuid": "corpus string", ....}
                relevant_docs: {"query_uuid": ["corpus_uuid"], ....}
            })
            evaluator_train: Dict({
                queries: {"query_uuid": "query string", .....}
                corpus: {"corpus_uuid": "corpus string", ....}
                relevant_docs: {"query_uuid": ["corpus_uuid"], ....}
            })
        })

    Train and Val contains only anchors (query) and positive (corpus) pairs that is needed for training and validation by the model.
    To calculate trining loss and validation loss.
    evaluator_val is used by the evaluator during the training process. 
    evaluator_test is used by evaluation function at the end of the training (InformationRetreivealEvaluator) which needs queries, corpus and relevant_docs data.
    evaluator_train is used by evalutation funtion when required by user to test the evaluation in training data.
"""
def print_dataset_info(datasets):
    # Train Dataset
    print(f"Train dataset size: {len(datasets['train'])}")

    # Val Dataset
    print(f"Validation dataset size: {len(datasets['val'])}")

    # Evaluator Val Dataset
    print(f"Evaluator Val dataset size: ")
    print(f"\t queries: {len(datasets['evaluator_val']['queries'])}")
    print(f"\t corpus: {len(datasets['evaluator_val']['corpus'])}")
    print(f"\t relevant_docs: {len(datasets['evaluator_val']['relevant_docs'])}")

    # Evaluator Test Dataset
    print(f"Evaluator Test dataset size: ")
    print(f"\t queries: {len(datasets['evaluator_test']['queries'])}")
    print(f"\t corpus: {len(datasets['evaluator_test']['corpus'])}")
    print(f"\t relevant_docs: {len(datasets['evaluator_test']['relevant_docs'])}")

def load_data(file_path: str) -> Dict:
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


# Create <anchor, positive> pair for training and validation of embedding model.
def create_train_val_dataset(data: Dict) -> Dataset:
    queries = data.get("queries", {})
    corpus = data.get("corpus", {})
    relevant_docs = data.get("relevant_docs", {})

    anchors = []
    positives = []

    for query_uuid, corpus_uuids in relevant_docs.items():
        query_text = queries.get(query_uuid, "")
        for corpus_uuid in corpus_uuids:
            corpus_text = corpus.get(corpus_uuid, "")
            if query_text and corpus_text:
                anchors.append(query_text)
                positives.append(corpus_text)

    return Dataset.from_dict(
        {
            "question": anchors,
            "answer": positives,
        }
    )

# Split the entire dataset into train, val and test
def split_data(
    data: Dict, train_ratio: float, val_ratio: float
) -> Tuple[Dict, Dict, Dict]:
    queries = data.get("queries", {})
    corpus = data.get("corpus", {})
    relevant_docs = data.get("relevant_docs", {})

    total_queries = list(queries.keys())

    # Shuffle the queries randomly
    random.seed(42)
    random.shuffle(total_queries)

    train_size = int(len(total_queries) * train_ratio)
    val_size = int(len(total_queries) * val_ratio)

    train_queries = total_queries[:train_size]
    val_queries = total_queries[train_size : train_size + val_size]
    test_queries = total_queries[train_size + val_size :]

    train_data = {
        "queries": {k: queries[k] for k in train_queries},
        "corpus": corpus,
        "relevant_docs": {k: relevant_docs[k] for k in train_queries},
    }

    val_data = {
        "queries": {k: queries[k] for k in val_queries},
        "corpus": corpus,
        "relevant_docs": {k: relevant_docs[k] for k in val_queries},
    }

    test_data = {
        "queries": {k: queries[k] for k in test_queries},
        "corpus": corpus,
        "relevant_docs": {k: relevant_docs[k] for k in test_queries},
    }

    return train_data, val_data, test_data

# Combine all the data into Train, val for training and Evaluator_val, Evaluator_test and Evaluator_train for evaluation
def get_datasets(
    file_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> DatasetDict:
    data = load_data(file_path)
    train_data, val_data, test_data = split_data(data, train_ratio, val_ratio)

    train_dataset = create_train_val_dataset(train_data)
    val_dataset = create_train_val_dataset(val_data)
    evaluator_val_dataset = DatasetDict(val_data)
    test_dataset = DatasetDict(test_data)
    evaluator_train_dataset = DatasetDict(train_data)

    return DatasetDict(
        {
            "train": train_dataset,
            "val": val_dataset,
            "evaluator_val": evaluator_val_dataset,
            "evaluator_test": test_dataset,
            "evaluator_train": evaluator_train_dataset,
        }
    )


# Example usage
# file_path = "path to the formatted dataset"
# datasets = get_datasets(file_path)

# Output the sizes of the datasets to verify
# print(f"Train dataset size: {len(datasets['train'])}")
# print(f"Validation dataset size: {len(datasets['validation'])}")
# print(f"Test dataset size: {len(datasets['test'])}")

# # Verify the dataset dictionary
# print(datasets)
