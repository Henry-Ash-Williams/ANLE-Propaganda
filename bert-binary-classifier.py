import warnings
from typing import Tuple, Literal, Optional

import pandas as pd
import torch
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.model_selection import train_test_split
import wandb

warnings.simplefilter(action="ignore", category=FutureWarning)

DATASET_PATH = (
    "/Users/henrywilliams/Documents/uni/anle/assessment/propaganda_dataset_v2"
)
TRAIN_DATASET = "propaganda_train.tsv"
VAL_DATASET = "propaganda_val.tsv"

BEGINNING_OF_SPAN = "<BOS>"
END_OF_SPAN = "<EOS>"

MODEL_NAME = "bert-base-uncased"

BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5
DECAY = 0.01
DROPOUT_RATE = 0.1

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

bert_config = BertConfig.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
)
bert_config.hidden_dropout_prob = DROPOUT_RATE
bert_config.attention_probs_dropout_prob = DROPOUT_RATE

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=bert_config,
)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
model.to(device)


def evaluate(
    dataloader: DataLoader, validation: bool = False
) -> Tuple[np.ndarray, np.int64, Optional[float], Optional[float]]:
    total_loss = 0 if validation else None
    total_acc = 0 if validation else None
    model.eval()
    predictions, true_labels = [], []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        label_ids = inputs["labels"].cpu().numpy()
        if validation:
            total_loss += loss.item()
            total_acc += accuracy_score(label_ids, logits.argmax(axis=1))

        predictions.append(logits)
        true_labels.append(label_ids)

    predictions = [item for sublist in predictions for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]
    return predictions, true_labels, total_loss, total_acc


def create_datasets() -> Tuple[DataLoader, DataLoader, DataLoader]:
    def remove_span_tags(sample: str) -> str:
        return sample.replace(BEGINNING_OF_SPAN, "").replace(END_OF_SPAN, "")

    def encode_dataset(df: pd.DataFrame) -> Tuple[any, any, any]:
        encoded = tokenizer.batch_encode_plus(
            df["tagged_in_context"],
            add_special_tokens=True,
            return_attention_mask=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return (
            encoded["input_ids"],
            encoded["attention_mask"],
            torch.tensor(df["label"].values),
        )

    def create_dataset_from_parts(
        inputs: torch.Tensor,
        masks: torch.Tensor,
        labels: torch.Tensor,
        sampler: Literal["random", "sequential"] = "random",
    ) -> DataLoader:
        dataset = TensorDataset(inputs, masks, labels)
        return DataLoader(
            dataset,
            sampler=(
                RandomSampler(dataset)
                if sampler == "random"
                else SequentialSampler(dataset)
            ),
            batch_size=BATCH_SIZE,
        )

    def convert_to_binary_label(label: str) -> int:
        return 1 if label != "not_propaganda" else 0

    train = pd.read_csv(
        f"{DATASET_PATH}/{TRAIN_DATASET}", sep="\t", header=0, quoting=3
    )
    val = pd.read_csv(f"{DATASET_PATH}/{VAL_DATASET}", sep="\t", header=0, quoting=3)

    train["tagged_in_context"] = train["tagged_in_context"].apply(remove_span_tags)
    val["tagged_in_context"] = val["tagged_in_context"].apply(remove_span_tags)

    train["label"] = train["label"].apply(convert_to_binary_label)
    val["label"] = val["label"].apply(convert_to_binary_label)

    val, test = train_test_split(val, test_size=0.3)

    train_inputs, train_masks, train_labels = encode_dataset(train)
    val_inputs, val_masks, val_labels = encode_dataset(val)
    test_inputs, test_masks, test_labels = encode_dataset(test)

    train_dataloader = create_dataset_from_parts(
        train_inputs, train_masks, train_labels, sampler="random"
    )
    validation_dataloader = create_dataset_from_parts(val_inputs, val_masks, val_labels)
    test_dataloader = create_dataset_from_parts(test_inputs, test_masks, test_labels)

    return train_dataloader, validation_dataloader, test_dataloader


def train(do_pretrain=False):
    train_dataloader, validation_dataloader, test_dataloader = create_datasets()

    if do_pretrain:
        print("Pre-training performance:")
        test_predictions, test_labels, _, _ = evaluate(test_dataloader)
        test_predictions = [np.argmax(pred) for pred in test_predictions]
        print(classification_report(test_predictions, test_labels))

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=DECAY)

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        train_iterator = tqdm(
            train_dataloader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"
        )

        for batch in train_iterator:
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
            }

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            wandb.log({"train": {"loss": loss.item()}})
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        wandb.log({"train": {"avg_loss": avg_train_loss}})

        val_predictions, val_labels, total_eval_loss, total_eval_accuracy = evaluate(
            validation_dataloader, validation=True
        )
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        val_predictions = [np.argmax(pred) for pred in val_predictions]
        metrics = classification_report(
            val_predictions, val_labels, target_names=["0", "1"], output_dict=True
        )
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        wandb.log({"val": {"acc": avg_val_accuracy}})
        wandb.log({"val": {"avg_loss": avg_val_loss}})
        wandb.log(
            {
                "val": {
                    "f1": metrics["weighted avg"]["f1-score"],
                    "precision": metrics["weighted avg"]["precision"],
                    "recall": metrics["weighted avg"]["recall"],
                }
            }
        )

    test_predictions, test_labels, _, _ = evaluate(test_dataloader)
    test_predictions = [np.argmax(pred) for pred in test_predictions]

    print(classification_report(test_labels, test_predictions))
    metrics = classification_report(
        test_labels, test_predictions, output_dict=True, target_names=["0", "1"]
    )
    model.save_pretrained(f"./models/{MODEL_NAME}-binary-propaganda-detection")
    return metrics


if __name__ == "__main__":
    wandb.login()
    wandb.init(
        project="propaganda-detection",
        notes="Binary propaganda classification with bert, including weight decay and dropout rate",
    )
    wandb.config = {
        "epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "dropout_rate": DROPOUT_RATE,
        "decay_rate": DECAY,
    }
    wandb.watch(model)
    test_metrics = train()
    wandb.log_artifact(f"./models/{MODEL_NAME}-binary-propaganda-detection")
    wandb.run.summary["test_accuracy"] = test_metrics["accuracy"]
    wandb.run.summary["test_precision"] = test_metrics["weighted avg"]["precision"]
    wandb.run.summary["test_recall"] = test_metrics["weighted avg"]["recall"]
    wandb.run.summary["test_f1"] = test_metrics["weighted avg"]["f1-score"]
    wandb.finish()
