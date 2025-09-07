import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

# Import your feature engineering functions
from featureEngineering import engineer_features  

def train_model(data_path, model_path="xgb_model.pkl"):
    # 1. Load raw data
    df = pd.read_csv(data_path)

    # 2. Apply feature engineering
    df = engineer_features(df)   # master function that applies all steps

    # 3. Define target and features
    target = "log_price"
    X = df.drop(columns=[target])  # drop target + unnecessary cols
    y = df[target]

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Define XGBoost model
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # 6. Train model
    model.fit(X_train, y_train)

    # 7. Evaluate model
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)  # ✅ modern version
    r2 = r2_score(y_test, y_pred)
    print(f"✅ Model trained! RMSE: {rmse:.2f}, R²: {r2:.2f}")

    # 8. Save model
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

    # 9. Save predictions to Excel (optional)
    results = X_test.copy()
    results["actual_price"] = y_test
    results["predicted_price"] = y_pred
    results.to_excel("predictions.xlsx", index=False)

    return model

if __name__ == "__main__":
    train_model(r"D:\Big Projects\Uber Dynamic price Detection\Data\rideshare_kaggle.csv")
