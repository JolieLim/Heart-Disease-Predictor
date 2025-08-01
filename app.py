import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# BMI calculation
def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)

# Streamlit input form
def show_user_form():
    st.title("💓 Heart Disease Predictor")
    st.markdown("### Please enter your details below:")

    # Inputs
    gender = st.selectbox("Sex", ["Female", "Male"])
    age_category = st.selectbox("Age Category", [
        '18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
        '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'
    ])
    race = st.selectbox("Race", [
        'White', 'Black', 'Asian', 'American Indian/Alaskan Native',
        'Hispanic', 'Other'
    ])
    smoking = st.selectbox("Have you smoked 100+ cigarettes?", ['No', 'Yes'])
    alcohol = st.selectbox("Heavy alcohol consumption?", ['No', 'Yes'])
    stroke = st.selectbox("Ever had a stroke?", ['No', 'Yes'])
    physical_activity = st.selectbox("Physical activity in past 30 days?", ['Yes', 'No'])
    diff_walking = st.selectbox("Difficulty walking/climbing stairs?", ['No', 'Yes'])
    diabetic = st.selectbox("Are you diabetic?", ['No', 'Yes', 'No, borderline diabetes', 'Yes (during pregnancy)'])
    physical_health = st.slider("Poor physical health days (last 30)", 0, 30, 0)
    mental_health = st.slider("Poor mental health days (last 30)", 0, 30, 0)
    sleep_time = st.slider("Average hours of sleep per day", 0, 24, 7)
    general_health = st.selectbox("General health rating", ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'])
    asthma = st.selectbox("Do you have asthma?", ['No', 'Yes'])
    kidney_disease = st.selectbox("Do you have kidney disease?", ['No', 'Yes'])
    skin_cancer = st.selectbox("Do you have skin cancer?", ['No', 'Yes'])
    height_cm = st.number_input("Height (cm)", 100, 250, 170)
    weight_kg = st.number_input("Weight (kg)", 30, 300, 70)

    bmi = calculate_bmi(weight_kg, height_cm)
    st.markdown(f"**Calculated BMI:** `{bmi}`")

    return {
        "Sex": gender,
        "AgeCategory": age_category,
        "Race": race,
        "Smoking": smoking,
        "AlcoholDrinking": alcohol,
        "Stroke": stroke,
        "PhysicalActivity": physical_activity,
        "DiffWalking": diff_walking,
        "Diabetic": diabetic,
        "PhysicalHealth": physical_health,
        "MentalHealth": mental_health,
        "SleepTime": sleep_time,
        "GenHealth": general_health,
        "Asthma": asthma,
        "KidneyDisease": kidney_disease,
        "SkinCancer": skin_cancer,
        "BMI": bmi
    }

# Encode categorical values
def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])

    # One-hot encode matching training preprocessing
    df_encoded = pd.get_dummies(df)

    # Get expected columns from scaler
    expected_cols = scaler.feature_names_in_
    for col in expected_cols:
        if col not in df_encoded:
            df_encoded[col] = 0

    df_encoded = df_encoded[expected_cols]
    return scaler.transform(df_encoded)

# Prediction function
def predict_heart_disease(form_data):
    X_scaled = preprocess_input(form_data)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]
    return prediction, probability

# --- Main execution ---
form_data = show_user_form()

if st.button("💡 Predict"):
    prediction, prob = predict_heart_disease(form_data)
    if prediction == 1:
        st.error(f"⚠️ High Risk: Likely to have heart disease.\nProbability: {prob:.2%}")
    else:
        st.success(f"✅ Low Risk: Not likely to have heart disease.\nProbability: {prob:.2%}")

st.markdown(    
    f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url('https://img.freepik.com/premium-photo/abstract-human-heart-with-pulse-line-ecg-wave-depicted-dark-blue-background-symbolizes-cardiac-health-medical-diagnostics_982248-7567.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    """, unsafe_allow_html=True
)