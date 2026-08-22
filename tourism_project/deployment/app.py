#%%writefile tourism_project/deployment/app.py
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"
ENCODERS_PATH = APP_DIR / "encoders.joblib"

# Categories the LabelEncoders were actually fit on. NOTE: prep.py renames
# "Single" -> "Unmarried" for MaritalStatus *before* fitting the encoder, so
# the encoder's classes_ never contain "Single" - the UI collects "Unmarried"
# directly to stay in sync with what was trained on.
CATEGORY_OPTIONS = {
    "TypeofContact": ["Self Enquiry", "Company Invited"],
    "Occupation": ["Salaried", "Small Business", "Large Business", "Free Lancer"],
    "Gender": ["Male", "Female"],
    "ProductPitched": ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
    "MaritalStatus": ["Unmarried", "Married", "Divorced"],
    "Designation": ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
}

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="🏨",
    layout="wide",
)


# ============================================================
# LOAD MODEL + ENCODERS
# ============================================================

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at `{MODEL_PATH}`. Run the training pipeline first.")
        st.stop()
    if not ENCODERS_PATH.exists():
        st.error(f"Encoders file not found at `{ENCODERS_PATH}`. Run the training pipeline first.")
        st.stop()
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, encoders


model, encoders = load_artifacts()


# ============================================================
# HEADER
# ============================================================

st.title("🏨 Tourism Package Prediction")
st.write(
    "Enter customer details below to predict whether the customer "
    "is likely to purchase the tourism package."
)
st.divider()


# ============================================================
# USER INPUTS
# ============================================================

with st.form("prediction_form"):
    st.subheader("Customer Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", CATEGORY_OPTIONS["Gender"])
        maritalstatus = st.selectbox("Marital Status", CATEGORY_OPTIONS["MaritalStatus"])
        occupation = st.selectbox("Occupation", CATEGORY_OPTIONS["Occupation"])
        designation = st.selectbox("Designation", CATEGORY_OPTIONS["Designation"])
        monthlyincome = st.number_input("Monthly Income", min_value=0, value=25000, step=1000)

    with col2:
        typeofcontact = st.selectbox("Type of Contact", CATEGORY_OPTIONS["TypeofContact"])
        citytier = st.selectbox("City Tier", [1, 2, 3])
        numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
        numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0, value=0)
        numberoftrips = st.number_input("Number of Trips (per year)", min_value=0, value=2)
        passport = st.selectbox("Has Passport?", ["Yes", "No"])

    with col3:
        productpitched = st.selectbox("Product Pitched", CATEGORY_OPTIONS["ProductPitched"])
        preferredpropertystar = st.selectbox("Preferred Property Star", [3, 4, 5])
        durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0, value=10)
        numberoffollowups = st.number_input("Number of Followups", min_value=0, value=3)
        pitchsatisfactionscore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
        owncar = st.selectbox("Owns a Car?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Package Purchase", type="primary", use_container_width=True)


# ============================================================
# PREDICTION
# ============================================================

if submitted:
    raw_input = {
        "Age": age,
        "TypeofContact": typeofcontact,
        "CityTier": citytier,
        "DurationOfPitch": durationofpitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": numberofpersonvisiting,
        "NumberOfFollowups": numberoffollowups,
        "ProductPitched": productpitched,
        "PreferredPropertyStar": preferredpropertystar,
        "MaritalStatus": maritalstatus,
        "NumberOfTrips": numberoftrips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitchsatisfactionscore,
        "OwnCar": 1 if owncar == "Yes" else 0,
        "NumberOfChildrenVisiting": numberofchildrenvisiting,
        "Designation": designation,
        "MonthlyIncome": monthlyincome,
    }

    input_df = pd.DataFrame([raw_input])

    # Apply the exact same LabelEncoders fit during training
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col].astype(str))

    # Match the exact column set/order the model was trained on
    expected_columns = model.feature_names_in_
    for column in expected_columns:
        if column not in input_df.columns:
            input_df[column] = 0
    input_df = input_df[expected_columns]

    prediction = model.predict(input_df)[0]

    st.divider()
    st.subheader("Prediction Result")

    result_col, prob_col = st.columns([2, 1])

    with result_col:
        if prediction == 1:
            st.success("🎉 This customer is **likely** to purchase the tourism package.")
        else:
            st.info("This customer is **unlikely** to purchase the tourism package.")

    with prob_col:
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]
            st.metric("Purchase Probability", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))

    with st.expander("View submitted customer details"):
        st.dataframe(pd.DataFrame([raw_input]), use_container_width=True)
