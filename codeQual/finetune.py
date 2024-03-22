from datasets import DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from codeQual import set_device


device = set_device.set()
num_classes = 3

# Load pre-trained model and tokenizer
model_name = "microsoft/codebert-base"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_classes, max_length=512
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})


# Prepare your data
def tokenize(batch):
    return tokenizer(batch["submission"], padding=True, truncation=True)


codequal: DatasetDict = DatasetDict.load_from_disk(
    "data/CodeQualData/code_qual_dataset"
)
codequal_encoded: DatasetDict = codequal.map(tokenize, batched=True, batch_size=None)


# Define evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# Create a Trainer
batch_size = 4
logging_steps = len(codequal_encoded["train"]) // batch_size
training_args = TrainingArguments(
    output_dir="training_output",
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=False,
    log_level="error",
    neftune_noise_alpha=0.1,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=codequal_encoded["train"],
    eval_dataset=codequal_encoded["validation"],
    tokenizer=tokenizer,
)
