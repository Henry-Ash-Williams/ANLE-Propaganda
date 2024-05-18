import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def remove_span_tags(sample: str):
    return sample.replace(BEGINNING_OF_SPAN, "").replace(END_OF_SPAN, "")


DATASET_PATH = "propaganda_dataset_v2"
TRAIN_DATASET = "propaganda_train.tsv"
VAL_DATASET = "propaganda_val.tsv"

BEGINNING_OF_SPAN = "<BOS>"
END_OF_SPAN = "<EOS>"

train = pd.read_csv(f"{DATASET_PATH}/{TRAIN_DATASET}", sep="\t", header=0, quoting=3)
val = pd.read_csv(f"{DATASET_PATH}/{VAL_DATASET}", sep="\t", header=0, quoting=3)

train["tagged_in_context"] = train["tagged_in_context"].apply(remove_span_tags)
val["tagged_in_context"] = val["tagged_in_context"].apply(remove_span_tags)

train["label"] = train["label"].apply(lambda l: 1 if l != "not_propaganda" else 0)
val["label"] = val["label"].apply(lambda l: 1 if l != "not_propaganda" else 0)

X_train, X_test, y_train, y_test = train_test_split(
    train["tagged_in_context"], train["label"], test_size=0.2
)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

overall_metrics = {
    "linear": {"precision": [], "recall": [], "f1": [], "acc": []},
    "poly": {"precision": [], "recall": [], "f1": [], "acc": []},
    "rbf": {"precision": [], "recall": [], "f1": [], "acc": []},
    "sigmoid": {"precision": [], "recall": [], "f1": [], "acc": []},
}

# for i in range(10):
# for kernel in ["linear", "poly", "rbf", "sigmoid"]:
svm = SVC(kernel="linear")
fit_start = time.time()
svm.fit(X_train_vec, y_train)
print(f"Fitting took {time.time() - fit_start}")

infer_start = time.time()
y_pred = svm.predict(X_test_vec)
print(f"Took {time.time() - infer_start} to make {len(y_pred)}")

# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
this_metrics = classification_report(y_test, y_pred, output_dict=True)
# overall_metrics[kernel]['precision'].append(this_metrics['weighted avg']['precision'])
# overall_metrics[kernel]['recall'].append(this_metrics['weighted avg']['recall'])
# overall_metrics[kernel]['f1'].append(this_metrics['weighted avg']['f1-score'])
# overall_metrics[kernel]['acc'].append(this_metrics['accuracy'])


for kernel, metrics in overall_metrics.items():
    print(f"====={kernel}===== ")
    print(f"Precision:\t{np.round( np.mean(metrics['precision']), decimals=2 )}")
    print(f"Recall:\t\t{np.round(np.mean(metrics['recall']), decimals=2)}")
    print(f"F1:\t\t{np.round(np.mean(metrics['f1']), decimals=2)}")
    print(f"Acc:\t\t{np.round(np.mean(metrics['acc']), decimals=2)}")
