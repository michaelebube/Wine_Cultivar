import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import json


def load_data(path="wine.csv"):
    df = pd.read_csv(path)
    # ensure target column exists
    if "Cultivars" not in df.columns:
        raise ValueError("Expected column 'Cultivars' in dataset")
    X = df.drop(columns=["Cultivars"])
    # drop unnamed index column if present
    X = X.loc[:, ~X.columns.str.contains("^Unnamed")]
    y = df["Cultivars"].astype(int)
    return X, y


def train_and_save(
    path="wine.csv", model_out="model.joblib", meta_out="model_meta.json"
):
    X, y = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    # Save model and metadata
    joblib.dump(clf, model_out)
    meta = {
        "features": list(X.columns),
        "classes": [int(c) for c in clf.classes_.tolist()],
        "class_names": {
            str(int(c)): f"Cultivar {int(c)}" for c in clf.classes_.tolist()
        },
    }
    with open(meta_out, "w") as f:
        json.dump(meta, f)

    print(f"Saved model to {model_out} and metadata to {meta_out}")


if __name__ == "__main__":
    train_and_save()
