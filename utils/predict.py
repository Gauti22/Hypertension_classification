# utils/predict.py
import joblib
model = joblib.load("/Users/gautammehta/Desktop/smal_project/model/final_model.pkl")

def predict_personality(features_scaled):
    prediction = model.predict(features_scaled)
    return "Yes" if prediction[0] == 0 else "No"
