# streamlit_app.py
import streamlit as st
from utils.preprocess import preprocess_input
from utils.predict import predict_personality
import pandas as pd

st.title("🧠 Personality Predictor")
st.subheader("Are you an Introvert or Extrovert?")

# Example Inputs (customize per your feature set)
#'Age', 'Salt_Intake', 'Stress_Score', 'BP_History', 'Sleep_Duration', 'BMI', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status'

"""


 0   Age               1985 non-null   int64  
 1   Salt_Intake       1985 non-null   float64
 2   Stress_Score      1985 non-null   int64  
 3   BP_History        1985 non-null   object 
 4   Sleep_Duration    1985 non-null   float64
 5   BMI               1985 non-null   float64
 6   Medication        1186 non-null   object 
 7   Family_History    1985 non-null   object 
 8   Exercise_Level    1985 non-null   object 
 9   Smoking_Status    1985 non-null   object 
 10  Has_Hypertension  1985 non-null   object 


"""

age = st.number_input("Age")
Salt_Intake = st.number_input("enter salt input")
Stress_Score = st.number_input("enter the stress score")
BP_History = st.selectbox("BP",['Normal','Hypertension','Prehypertension'])
Sleep_Duration =st.number_input('sleep duration')
BMI = st.number_input('enter the bmi')
medication = st.selectbox("Medication", ["ACE Inhibitor","NaN","Other"])
family_history = st.selectbox("Family History", ["No", "Yes"])
exercise_level = st.selectbox("Exercise Level", ["Moderate", "Moderate", "High"])
smoking_status = st.selectbox("Smoking Status", ["Non-smoker", "smoker"])

# Map inputs to model features
user_input = {

    'Age': age,
    'Salt_Intake': Salt_Intake,
    'Stress_Score':Stress_Score,
    'BP_History':BP_History,
    'Sleep_Duration':Sleep_Duration,
    'BMI': BMI,
    'Medication' : medication,
    'Family_History': family_history,
    'Exercise_Level':exercise_level,
    'Smoking_Status':smoking_status
}

if st.button("Predict"):
    try:
        processed = preprocess_input(user_input)
        print(processed)
        result = predict_personality(processed)
        print(result)
        st.success(f"🎯 Prediction: **{result}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
