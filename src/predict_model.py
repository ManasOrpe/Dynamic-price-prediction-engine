
import joblib
import numpy as np
import pandas as pd

def load_model(path: str):
    """Load the trained model from disk."""
    return joblib.load(path)

def predict_dataframe(model, X: pd.DataFrame) -> np.ndarray:
    """
    Predict with a DataFrame that already matches training features.
    Returns raw model outputs (your model was trained on log_price or price?).
    If it was trained on log_price, you may want to np.expm1() here.
    """
    y_pred = model.predict(X)
    # If your target during training was log_price, uncomment the next line:
    # y_pred = np.expm1(y_pred)
    return y_pred
