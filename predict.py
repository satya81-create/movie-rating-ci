import joblib
import pandas as pd

def predict_rating(genre, duration, budget, popularity):
    model = joblib.load("model.pkl")
    data = pd.DataFrame([{
        "Genre": genre,
        "Duration_Min": duration,
        "Budget_Crore_INR": budget,
        "Popularity_Index": popularity,
    }])
    return float(model.predict(data)[0])

if __name__ == "__main__":
    print("Predicted rating:", predict_rating("Drama", 130, 50.0, 75.0))