import warnings
import argparse
from typing import Tuple, Literal, Optional, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import wandb
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)

argparser = argparse.ArgumentParser()

argparser.add_argument("--learning_rate", type=float, default=0.01057728415157835)
argparser.add_argument("--batch_size", type=int, default=32)
argparser.add_argument("--dropout_rate", type=float, default=0.18604433133268403)
argparser.add_argument("--weight_decay", type=float, default=0.01057728415157835)
argparser.add_argument("--epochs", type=int, default=10)
argparser.add_argument("--optimizer", type=str, default="SGD")

args = argparser.parse_args()

DATASET_PATH = (
    "../propaganda_dataset_v2/"
)
TRAIN_DATASET = "propaganda_train.tsv"
VAL_DATASET = "propaganda_val.tsv"

BEGINNING_OF_SPAN = "<BOS>"
END_OF_SPAN = "<EOS>"

MODEL_NAME = "bert-base-uncased"

LR = args.learning_rate 
BATCH_SIZE = args.batch_size 
DROPOUT_RATE = args.dropout_rate 
DECAY = args.weight_decay 
EPOCHS = args.epochs 
OPTIMIZER = args.optimizer 

LABELS = [
    "not_propaganda",
    "propaganda",
]

NUM_LABELS = len(LABELS)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

bert_config = BertConfig.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
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
    dataloader: DataLoader, record_total_loss: bool = False
) -> Tuple[np.ndarray, np.int64, Optional[float]]:
    """
    Evaluate the model on the provided dataset.

    Args:
        dataset (DataLoader): DataLoader containing the evaluation dataset.
        record_total_loss (bool, optional): Whether to record the total loss.
            Defaults to False.

    Returns:
        Tuple[List[np.ndarray], List[np.int32], Optional[int]]: A tuple containing:
            - predictions (List[np.ndarray]): Predicted logits for each sample.
            - true_labels (List[np.int32]): True labels for each sample.
            - total_loss (Optional[int]): Total loss if `record_total_loss` is True,
              otherwise None.
    """
    total_loss = 0 if record_total_loss else None
    model.eval()
    predictions, true_labels = [], []

    for input_ids, attention_mask, labels in dataloader:
        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels.to(device),
        }

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        label_ids = inputs["labels"].cpu().numpy()

        if record_total_loss:
            total_loss += loss.item()

        predictions.append(logits)
        true_labels.append(label_ids)

    predictions = [
        np.argmax(prediction)
        for batched_predictions in predictions
        for prediction in batched_predictions
    ]
    true_labels = [label for batched_labels in true_labels for label in batched_labels]
    return predictions, true_labels, total_loss


def test_evaluation(dataloader: DataLoader) -> Tuple[List[np.int32], List[np.int32]]:
    """
    Evaluate the model using the provided dataloader.

    Args:
        dataloader (DataLoader): DataLoader object for test/validation data.

    Returns:
        Tuple[List[np.int32], List[np.int32]]: Predicted labels and true labels.
    """
    evaluation = evaluate(dataloader)
    return evaluation[0], evaluation[1]


def validation(dataloader: DataLoader) -> Tuple[List[np.int32], List[np.int32], int]:
    """
    Validate the model using the provided dataloader and record total loss.

    Args:
        dataloader (DataLoader): DataLoader object for validation data.

    Returns:
        Tuple[List[np.int32], List[np.int32], int]: Predicted labels, true labels, and total evaluation loss.
    """
    evaluation = evaluate(dataloader, record_total_loss=True)
    return evaluation[0], evaluation[1], evaluation[2]


def remove_span_tags(sample: str) -> str:
    """
    Remove <BOS> and <EOS> tags from the input sample.

    Args:
        sample (str): Input text sample.

    Returns:
        str: Text sample with <BOS> and <EOS> tags removed.
    """
    return sample.replace(BEGINNING_OF_SPAN, "").replace(END_OF_SPAN, "")


def encode_dataset(df: pd.DataFrame) -> Tuple[any, any, any]:
    """
    Encode the dataset using the tokenizer.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing input_ids, attention_mask, and labels.
    """
    encoded = tokenizer.batch_encode_plus(
        df["tagged_in_context"],
        add_special_tokens=True,
        return_attention_mask=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return (
        encoded["input_ids"].to(torch.long),
        encoded["attention_mask"].to(torch.long),
        torch.tensor(df["label"].values, dtype=torch.long),
    )


def create_dataset_from_parts(
    inputs: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    sampler: Literal["random", "sequential"] = "random",
) -> DataLoader:
    """
    Remove <BOS> and <EOS> tags from the input sample.

    Args:
        sample (str): Input text sample.

    Returns:
        str: Text sample with <BOS> and <EOS> tags removed.
    """
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
    """
    Convert label to binary representation.

    Args:
        label (str): Input label.

    Returns:
        int: Binary representation of the label (1 for propaganda, 0 for not propaganda).
    """
    return 1 if label != "not_propaganda" else 0


def create_datasets() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convert label to binary representation.

    Args:
        label (str): Input label.

    Returns:
        int: Binary representation of the label (1 for propaganda, 0 for not propaganda).
    """
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


def get_optimizer(optimizer_name: str) -> Optimizer:
    """
    Get an instance of the specified optimizer.

    Args:
        optimizer_name (str): The name of the optimizer class.

    Returns:
        torch.optim.Optimizer: An instance of the specified optimizer class.

    Example:
        optimizer = get_optimizer('Adam')
    """
    optimizer_cls = getattr(optim, optimizer_name)
    return optimizer_cls(model.parameters(), lr=LR, weight_decay=DECAY)


def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plot the confusion matrix and save it to disk

    Args:
        y_true (List): True labels.
        y_pred (List): Predicted labels.
        labels (List[str]): List of class labels.
    """
    plt.ioff()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(
        f"./models/{MODEL_NAME}-binary-propaganda-detection/classification_confusion_matrix.png"
    )


def train() -> dict:
    """
    Train the BERT model for binary propaganda detection.

    Returns:
        dict: Classification report including precision, recall, F1-score, and accuracy.
    """
    train_dataloader, validation_dataloader, test_dataloader = create_datasets()

    optimizer = get_optimizer(OPTIMIZER)

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        train_iterator = tqdm(
            train_dataloader, desc=f"Training Epoch {epoch + 1:02}/{EPOCHS}"
        )

        for input_ids, attention_mask, labels in train_iterator:
            inputs = {
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device),
                "labels": labels.to(device),
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

        val_predictions, val_labels, total_val_loss = validation(validation_dataloader)
        val_metrics = classification_report(
            val_predictions, val_labels, target_names=LABELS, output_dict=True, zero_division=0.0
        )
        avg_val_loss = total_val_loss / len(validation_dataloader)
        wandb.log(
            {
                "val": {
                    "f1": val_metrics["weighted avg"]["f1-score"],
                    "precision": val_metrics["weighted avg"]["precision"],
                    "recall": val_metrics["weighted avg"]["recall"],
                    "loss": avg_val_loss,
                    "acc": val_metrics["accuracy"],
                }
            }
        )

    test_predictions, test_labels = test_evaluation(test_dataloader)

    test_predictions = [LABELS[label] for label in test_predictions]
    test_labels = [LABELS[label] for label in test_labels]

    test_metrics = classification_report(
        test_labels, test_predictions, output_dict=True, target_names=LABELS
    )
    model.save_pretrained(f"./models/{MODEL_NAME}-binary-propaganda-detection")
    plot_confusion_matrix(test_labels, test_predictions, LABELS)
    return test_metrics


if __name__ == "__main__":
    wandb.login()
    wandb.init(
        project="propaganda-detection",
        notes="Baseline BERT Binary propaganda detection",
    )
    wandb.config = {
        "epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "dropout_rate": DROPOUT_RATE,
        "decay_rate": DECAY,
        "optimizer": OPTIMIZER,
    }
    wandb.watch(model)
    test_metrics = train()
    wandb.run.summary["test_accuracy"] = test_metrics["accuracy"]
    wandb.run.summary["test_precision"] = test_metrics["weighted avg"]["precision"]
    wandb.run.summary["test_recall"] = test_metrics["weighted avg"]["recall"]
    wandb.run.summary["test_f1"] = test_metrics["weighted avg"]["f1-score"]
    wandb.log(
        {
            "test": {
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["weighted avg"]["precision"],
                "recall": test_metrics["weighted avg"]["recall"],
                "f1": test_metrics["weighted avg"]["f1-score"],
            }
        }
    )
    wandb.finish()
