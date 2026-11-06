import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "data/movies.csv"
CAT_COLS = ["Genre"]
NUM_COLS = ["Duration_Min", "Budget_Crore_INR", "Popularity_Index"]
TARGET = "Rating_10"

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print("Dataset shape:", df.shape)
print("\nMissing values before handling:")
print(df[CAT_COLS + NUM_COLS + [TARGET]].isnull().sum())

df = df.dropna(subset=[TARGET])
X = df[CAT_COLS + NUM_COLS]
y = df[TARGET]

preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]), CAT_COLS),
])

model = Pipeline([
    ("prep", preprocess),
    ("reg", RandomForestRegressor(n_estimators=100, random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("\nRMSE:", round(mean_squared_error(y_test, pred) ** 0.5, 4))
print("R2  :", round(r2_score(y_test, pred), 4))

joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")