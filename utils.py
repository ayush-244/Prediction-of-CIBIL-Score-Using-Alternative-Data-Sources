def predict_score(model, user_data):
    prediction = model.predict([user_data])[0]
    return prediction