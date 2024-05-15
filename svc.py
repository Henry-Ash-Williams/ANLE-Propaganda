import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def remove_span_tags(sample: str):
    return sample.replace(BEGINNING_OF_SPAN, "").replace(END_OF_SPAN, "")


DATASET_PATH = (
    "/Users/henrywilliams/Documents/uni/anle/assessment/propaganda_dataset_v2"
)
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

for kernel in ["linear", "poly", "rbf", "sigmoid"]:
    print(f"using kernel: {kernel}")
    svm = SVC(kernel=kernel)
    svm.fit(X_train_vec, y_train)

    y_pred = svm.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    # print("Classification Report:\n", classification_report(y_test, y_pred))
