import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from config import MODEL_PATH


def train_model(df):
    X = df.drop(columns=['shadow_score'])
    #y = df['shadow_score'] * 6 + 300   return CIBIL-like : 300–900
    y = df['shadow_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("RMSE:", root_mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)


def load_model():
    return joblib.load(MODEL_PATH)