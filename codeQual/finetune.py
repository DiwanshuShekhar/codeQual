import os
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from codeQual import set_device, ROOT_DIR


device = set_device.set()
num_classes = 3

# Load pre-trained model and tokenizer
# https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForSequenceClassification
# bigcode/starcoder2-3b,
# microsoft/codebert-base, microsoft/phi-2,
# codellama/CodeLlama-7b-Python-hf, Salesforce/codet5p-220m-py,
model_name = os.getenv("MODEL_NAME")


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes, max_length=512
    ).to(device)


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})


# Prepare your data
def tokenize(batch):
    return tokenizer(batch["submission"], padding=True, truncation=True)


codequal: DatasetDict = DatasetDict.load_from_disk(
    os.path.join(ROOT_DIR, "data/hf_code_qual_dataset_v1")
)
codequal_encoded: DatasetDict = codequal.map(tokenize, batched=True, batch_size=None)


# Define evaluation metrics
def compute_metrics(pred):
    print(f"Input to compute metrics: {pred}")
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def get_trainer(
    run_name: str, epochs: int = 3, lr: float = 5e-5, batch_size: int = 4
) -> Trainer:
    """
    https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/trainer#transformers.TrainingArguments
    """
    training_args = TrainingArguments(
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.2,
        output_dir=f"{ROOT_DIR}/experiments/{run_name}",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",  # no, steps, epoch
        eval_steps=50,  # how often to evaluate on the validation set
        save_strategy="epoch",  # no, steps, epoch
        save_steps=50,  # how often to save the model
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        logging_dir=f"{ROOT_DIR}/experiments/{run_name}/logs",
        logging_steps=100,  # how often to log to W&B
        report_to="wandb",
        run_name=run_name,  # name of the W&B run (optional)
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=codequal_encoded["train"],
        eval_dataset=codequal_encoded["validation"],
        compute_metrics=compute_metrics,
    )

    return trainer
