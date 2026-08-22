import os
import joblib
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(_file_))
MODEL_PATH = os.path.join(APP_DIR, "model.joblib")
ENCODERS_PATH = os.path.join(APP_DIR, "encoders.joblib")

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)

st.title("Wellness Tourism Package Prediction")
st.write(
    "Enter customer and interaction details to predict whether they are "
    "likely to purchase the Wellness Tourism Package."
)

st.header("Customer Details")
age = st.number_input("Age", min_value=18, max_value=100, value=35)
type_of_contact = st.selectbox("Type of Contact", encoders["TypeofContact"].classes_)
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", encoders["Occupation"].classes_)
gender = st.selectbox("Gender", encoders["Gender"].classes_)
num_persons_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", encoders["MaritalStatus"].classes_)
num_trips = st.number_input("Number of Trips (per year)", min_value=0, max_value=20, value=2)
passport = st.selectbox("Holds Passport", ["No", "Yes"])
own_car = st.selectbox("Owns Car", ["No", "Yes"])
num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", encoders["Designation"].classes_)
monthly_income = st.number_input("Monthly Income", min_value=0, value=20000)

st.header("Customer Interaction Data")
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
product_pitched = st.selectbox("Product Pitched", encoders["ProductPitched"].classes_)
num_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=3)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=180, value=15)

if st.button("Predict"):
    raw_input = {
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_persons_visiting,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "ProductPitched": product_pitched,
        "NumberOfFollowups": num_followups,
        "DurationOfPitch": duration_of_pitch,
    }
    input_df = pd.DataFrame([raw_input])

    # Apply the same label encoding used during training
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col].astype(str))

    # Match the exact column order the model was trained on
    input_df = input_df.reindex(columns=model.feature_names_in_)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Likely to purchase the package (confidence: {probability:.1%})")
    else:
        st.info(f"Not likely to purchase the package (confidence: {1 - probability:.1%})")
