from predict import predict_rating

def test_prediction_returns_number():
    result = predict_rating("Drama", 130, 50.0, 75.0)
    assert isinstance(result, float)

def test_prediction_in_valid_range():
    result = predict_rating("Comedy", 110, 20.0, 40.0)
    assert 0 <= result <= 10