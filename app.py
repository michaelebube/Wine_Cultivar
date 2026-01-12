from flask import Flask, request, jsonify, send_from_directory
import joblib
import json
import os

APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_DIR, "model.joblib")
META_PATH = os.path.join(APP_DIR, "model_meta.json")

app = Flask(__name__, static_folder=APP_DIR)


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError(
            "Model or metadata not found. Run train_model.py first."
        )
    clf = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return clf, meta


clf, meta = None, None
try:
    clf, meta = load_model()
except Exception:
    # model may not exist during development; lazily load in endpoint
    clf, meta = None, None


@app.route("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global clf, meta
    if clf is None or meta is None:
        clf, meta = load_model()

    data = request.get_json() or request.form.to_dict()
    # assemble feature vector in order
    features = meta["features"]
    try:
        vals = [float(data.get(f, 0)) for f in features]
    except Exception as e:
        return jsonify({"error": f"Invalid input values: {e}"}), 400

    import numpy as np

    X = np.array(vals).reshape(1, -1)
    pred = clf.predict(X)[0]
    probs = clf.predict_proba(X)[0]

    # map classes to names
    class_order = [int(c) for c in clf.classes_]
    class_names = meta.get(
        "class_names", {str(c): f"Cultivar {c}" for c in class_order}
    )
    probs_map = {
        class_names[str(cls)]: float(probs[i]) for i, cls in enumerate(class_order)
    }

    return jsonify(
        {
            "predicted_label": int(pred),
            "predicted_name": class_names.get(str(int(pred)), f"Cultivar {int(pred)}"),
            "probabilities": probs_map,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
