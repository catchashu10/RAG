from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from dataset_layer.dataset_load import load_dataset_from_huggingface


def fine_tune_embedding():
    # Set parameters

    batch_size = 16
    epoch = 5
    model_name = "all-MiniLM-L6-v2"
    checkpoint_enable = False
    print(f"Checkpoint: {checkpoint_enable}")
    # model name : ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-mpnet-base-dot-v1", "all-distilroberta-v1", "all-MiniLM-L12-v2", "multi-qa-distilbert-cos-v1", "multi-qa-MiniLM-L6-cos-v1"]
    model_output_folder = "fine_tuned_models"
    mode_output_name = f"{model_name}_fineTuned_b{batch_size}_e{epoch}_bf16"


    # 1. Load a model to finetune:
    model = SentenceTransformer(model_name)
    dataset, test_ds, corpus = load_dataset_from_huggingface()
    
    # 3. Define a loss function
    loss = MultipleNegativesRankingLoss(model)
    
    # 4. Specify training arguments
    train_args = SentenceTransformerTrainingArguments (
        # Required parameter:
        output_dir=f"{model_output_folder}/{mode_output_name}",
        # Optional training parameters:
        num_train_epochs=epoch,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
        # Optional tracking/debugging parameters:
        evaluation_strategy="no",
        # eval_steps=100,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        logging_steps=50,
        log_on_each_node=False,
        run_name=mode_output_name,  # Will be used in W&B if `wandb` is installed
    )
    
    # 5. Create an evaluator & evaluate the base model
    eval_dataset = dataset["evaluator_train"]
    dev_evaluator = InformationRetrievalEvaluator(
        eval_dataset["queries"], eval_dataset["corpus"], eval_dataset["relevant_docs"]
    )

    print(f"Evaluation for Base model {model_name}\n: {dev_evaluator(model)}")
    
    # 6. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=test_ds,
        # eval_dataset=datasets["val"], # To get validation loss details during training. May slow down trainig.
        loss=loss,
        # evaluator=dev_evaluator, # To get detailed evaluation results during trainig.
    )
    trainer.train(resume_from_checkpoint=checkpoint_enable)
    
    # 7. Evaluate the trained model on the test set
    dev_evaluator(model)
    print(f"Evaluation for FineTuned model {model_name}\n: {dev_evaluator(model)}")
    
    # 8. Save the trained model
    model.save_pretrained(
        f"{model_output_folder}/{mode_output_name}"
    )

