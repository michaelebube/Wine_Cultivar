# Wine Cultivar Classification (Multi-Class)

This project trains a RandomForest classifier on the UCI Wine dataset to predict the wine cultivar (three classes).

Quick steps (local):

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Train the model (this will create `model.joblib` and `model_meta.json`):

```
python train_model.py
```

3. Run the Flask app:

```
python app.py
```

Open http://localhost:5000 in your browser and use the form to make predictions.

## Google Colab

To train in Google Colab: upload `wine.csv` into the Colab session or mount your Drive and run the same `train_model.py` cells. Example snippet for a Colab cell:

```python
!pip install -q -r requirements.txt
from train_model import train_and_save
train_and_save('wine.csv')
```

Notes

- The model handles multi-class classification natively.
- The app returns both the predicted label and per-class probabilities.
