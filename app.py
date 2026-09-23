
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and preprocessor
model = joblib.load("model_building/random_forest_model.pkl")
preprocessor = joblib.load("model_building/preprocessor.pkl")

# Page configuration
st.set_page_config(
    page_title="Wellness Tourism Package Prediction",
    page_icon="🌴",
    layout="centered"
)

st.title("🌴 Wellness Tourism Package Prediction")
st.write("Enter customer details to predict whether the customer is likely to purchase the Wellness Tourism Package.")

# Customer inputs

Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Married", "Unmarried", "Divorced"]
)

NumberOfTrips = st.number_input(
    "Number of Trips",
    min_value=0,
    max_value=30,
    value=2
)

Passport = st.selectbox(
    "Passport",
    [0, 1]
)

OwnCar = st.selectbox(
    "Own Car",
    [0, 1]
)

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)

Designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=0,
    value=25000
)

PitchSatisfactionScore = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

ProductPitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Deluxe",
        "Standard",
        "Super Deluxe",
        "King"
    ]
)

NumberOfFollowups = st.number_input(
    "Number of Followups",
    min_value=0,
    max_value=10,
    value=3
)

DurationOfPitch = st.number_input(
    "Duration of Pitch",
    min_value=0,
    max_value=120,
    value=15
)

# Create input dataframe
input_data = pd.DataFrame({
    "Age": [Age],
    "TypeofContact": [TypeofContact],
    "CityTier": [CityTier],
    "Occupation": [Occupation],
    "Gender": [Gender],
    "NumberOfPersonVisiting": [NumberOfPersonVisiting],
    "PreferredPropertyStar": [PreferredPropertyStar],
    "MaritalStatus": [MaritalStatus],
    "NumberOfTrips": [NumberOfTrips],
    "Passport": [Passport],
    "OwnCar": [OwnCar],
    "NumberOfChildrenVisiting": [NumberOfChildrenVisiting],
    "Designation": [Designation],
    "MonthlyIncome": [MonthlyIncome],
    "PitchSatisfactionScore": [PitchSatisfactionScore],
    "ProductPitched": [ProductPitched],
    "NumberOfFollowups": [NumberOfFollowups],
    "DurationOfPitch": [DurationOfPitch]
})

# Prediction
if st.button("Predict"):

    processed_data = preprocessor.transform(input_data)

    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    if prediction == 1:
        st.success("The customer is predicted to purchase the Wellness Tourism Package.")
    else:
        st.info("The customer is predicted not to purchase the Wellness Tourism Package.")

    st.write(f"Purchase Probability: {probability:.2%}")
