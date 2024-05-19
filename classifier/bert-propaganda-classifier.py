from typing import Tuple, Literal, List, Optional
import warnings
import argparse

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import torch.optim as optim
from torch.optim import Optimizer
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from tqdm import tqdm
import wandb

warnings.simplefilter(action="ignore", category=FutureWarning)

argparser = argparse.ArgumentParser()

argparser.add_argument("--learning_rate", type=float, default=2e-5)
argparser.add_argument("--batch_size", type=int, default=32)
argparser.add_argument("--dropout_rate", type=float, default=0.1)
argparser.add_argument("--weight_decay", type=float, default=0.01)
argparser.add_argument("--epochs", type=int, default=10)
argparser.add_argument("--optimizer", type=str, default="AdamW")

args = argparser.parse_args()

GENERATE_CONFUSION_MATRIX = True

DATASET_PATH = "../propaganda_dataset_v2"
TRAIN_DATASET = "propaganda_train.tsv"
VAL_DATASET = "propaganda_val.tsv"

BEGINNING_OF_SPAN = "<BOS>"
END_OF_SPAN = "<EOS>"

LABELS = [
    "flag_waving",
    "appeal_to_fear_prejudice",
    "causal_simplification",
    "doubt",
    "exaggeration,minimisation",
    "loaded_language",
    "name_calling,labeling",
    "repetition",
]


MODEL_NAME = "bert-base-uncased"

LR = args.learning_rate
BATCH_SIZE = args.batch_size
DROPOUT_RATE = args.dropout_rate
DECAY = args.weight_decay
EPOCHS = args.epochs
OPTIMIZER = args.optimizer

NUM_LABELS = len(LABELS)

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

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
model.to(device)
label_encoder = LabelEncoder()


def evaluate(
    dataset: DataLoader, record_total_loss: bool = False
) -> Tuple[List[np.ndarray], List[np.int32], Optional[int]]:
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
    model.eval()
    total_loss = 0 if record_total_loss else None
    predictions = []
    true_labels = []

    for input_ids, attention_mask, labels in dataset:
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
        Tuple[List[np.int32], List[np.int32]]:
            Predicted labels and true labels.
    """
    evaluation = evaluate(dataloader)
    return evaluation[0], evaluation[1]


def validation(dataloader: DataLoader) -> Tuple[List[np.int32], List[np.int32], int]:
    """
    Validate the model using the provided dataloader and record total loss.

    Args:
        dataloader (DataLoader): DataLoader object for validation data.

    Returns:
        Tuple[List[np.int32], List[np.int32], int]:
            Predicted labels, true labels, and total evaluation loss.
    """
    evaluation = evaluate(dataloader, record_total_loss=True)
    return evaluation[0], evaluation[1], evaluation[2]


def create_dataset_from_parts(
    inputs: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    sampler: Literal["random", "sequential"] = "random",
) -> DataLoader:
    """
    Create a PyTorch DataLoader from input tensors.

    Args:
        inputs (torch.Tensor): Tensor containing input data.
        masks (torch.Tensor): Tensor containing attention masks.
        labels (torch.Tensor): Tensor containing labels.
        sampler ({'random', 'sequential'}, optional): Sampler type.
            Defaults to 'random'.

    Returns:
        DataLoader: PyTorch DataLoader.
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


def encode_dataset(df: pd.DataFrame) -> Tuple[any, any, any]:
    """
    Encode the dataset using the tokenizer.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Tuple containing input_ids, attention_mask, and labels.
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


def extract_snippet(sample: str) -> str:
    """
    Extract text within the <BOS> and <EOS> tags.

    Args:
        sample (str): A text sample containing both <BOS> and <EOS>

    Returns:
        str: A string with only the text within the span
    """

    assert (
        BEGINNING_OF_SPAN in sample
    ), "Text sample should contain beginning of span tag (<BOS>)"
    assert END_OF_SPAN in sample, "Text sample should contain end of span tag (<EOS>)"
    s_idx = sample.index(BEGINNING_OF_SPAN) + len(BEGINNING_OF_SPAN)
    e_idx = sample.index(END_OF_SPAN)
    return sample[s_idx:e_idx]


def create_datasets() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare training, validation, and test datasets.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]:
            Tuple containing training, testing, and validation dataloaders.
    """
    train = pd.read_csv(
        f"{DATASET_PATH}/{TRAIN_DATASET}", sep="\t", quoting=3, header=0
    )
    val = pd.read_csv(f"{DATASET_PATH}/{VAL_DATASET}", sep="\t", quoting=3, header=0)

    train = train[train["label"] != "not_propaganda"]
    val = val[val["label"] != "not_propaganda"]

    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)

    train["tagged_in_context"] = train["tagged_in_context"].apply(extract_snippet)
    val["tagged_in_context"] = val["tagged_in_context"].apply(extract_snippet)
    train["label"] = label_encoder.fit_transform(train["label"])
    val["label"] = label_encoder.fit_transform(val["label"])
    val, test = train_test_split(val, test_size=0.3)

    train_inputs, train_masks, train_labels = encode_dataset(train)
    val_inputs, val_masks, val_labels = encode_dataset(val)
    test_inputs, test_masks, test_labels = encode_dataset(test)

    train_dataloader = create_dataset_from_parts(
        train_inputs, train_masks, train_labels, sampler="random"
    )
    validation_dataloader = create_dataset_from_parts(val_inputs, val_masks, val_labels)
    test_dataloader = create_dataset_from_parts(test_inputs, test_masks, test_labels)

    return train_dataloader, test_dataloader, validation_dataloader


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
    plt.figure()
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(
        f"./models/{MODEL_NAME}-propaganda-classification/classification_confusion_matrix.png"
    )


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
    try:
        optimizer_cls = getattr(optim, optimizer_name)
    except AttributeError:
        print(f"Could not find a pytorch optimizer with the name {optimizer_name}")
    return optimizer_cls(model.parameters(), lr=LR, weight_decay=DECAY)


def train():
    train, test, val = create_datasets()

    optimizer = get_optimizer(OPTIMIZER)

    model.train()

    for epoch in range(EPOCHS):
        total_training_loss = 0
        train_iterator = tqdm(train, desc=f"Training Epoch {epoch + 1:02}/{EPOCHS}")

        for input_ids, attention_mask, labels in train_iterator:
            inputs = {
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device),
                "labels": labels.to(device),
            }

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            total_training_loss += loss.item()
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss.item()})
        wandb.log({"avg_loss": total_training_loss / len(train)})
        val_predictions, val_labels, total_val_loss = validation(val)
        val_metrics = classification_report(
            val_labels,
            val_predictions,
            output_dict=True,
            target_names=LABELS,
            zero_division=0.0,
        )
        avg_val_loss = total_val_loss / len(val)
        wandb.log(
            {
                "val": {
                    "acc": val_metrics["accuracy"],
                    "loss": avg_val_loss,
                    "precision": val_metrics["weighted avg"]["precision"],
                    "recall": val_metrics["weighted avg"]["recall"],
                    "f1-score": val_metrics["weighted avg"]["f1-score"],
                }
            }
        )

    test_predictions, test_labels = test_evaluation(test)
    test_metrics = classification_report(
        test_labels,
        test_predictions,
        output_dict=True,
        target_names=LABELS,
        zero_division=0.0,
    )
    model.save_pretrained(f"./models/{MODEL_NAME}-propaganda-classification")

    if GENERATE_CONFUSION_MATRIX:
        test_labels = [
            label_encoder.inverse_transform([label])[0] for label in test_labels
        ]
        test_predictions = [
            label_encoder.inverse_transform([prediction])[0]
            for prediction in test_predictions
        ]
        plot_confusion_matrix(test_labels, test_predictions, labels=LABELS)
        wandb.log(
            {
                "classification_confusion_matrix": wandb.Image(
                    f"models/{MODEL_NAME}-propaganda-classification/classification_confusion_matrix.png"
                )
            }
        )

    return test_metrics


if __name__ == "__main__":
    wandb.login()
    wandb.init(
        project="propaganda-classification",
        notes="Baseline propaganda classification with BERT",
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
