import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

CSV_PATH = "wine.csv"
MODEL_PATH = "model.joblib"
META_PATH = "model_meta.json"
TARGET_COL = "Cultivars"

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()  # IMPORTANT

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(clf, MODEL_PATH)

meta = {
    "features": list(X.columns),
    "class_names": {"1": "Cultivar 1", "2": "Cultivar 2", "3": "Cultivar 3"},
}

with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print("Saved model and metadata successfully.")
