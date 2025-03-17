import joblib
import pandas as pd
import streamlit as st

model = joblib.load('kidney_disease_model.pkl')

def get_user_input():
    age = st.number_input("Enter age", min_value=1, max_value=100, value=25)
    blood_pressure = st.number_input("Enter blood pressure", min_value=50, max_value=200, value=120)
    specific_gravity = st.number_input("Enter specific gravity", min_value=1.005, max_value=1.025,format="%.3f", value=1.015)
    albumin = st.number_input("Enter albumin", min_value=0, max_value=5, value=0)
    sugar = st.number_input("Enter sugar", min_value=0, max_value=5, value=0)
    red_blood_cells = st.selectbox("Enter red blood cells (0 for abnormal, 1 for normal)", [0, 1])
    pus_cell = st.selectbox("Enter pus cell (0 for abnormal, 1 for normal)", [0, 1])
    pus_cell_clumps = st.selectbox("Enter pus cell clumps (0 for not present, 1 for present)", [0, 1])
    bacteria = st.selectbox("Enter bacteria (0 for not present, 1 for present)", [0, 1])
    blood_glucose_random = st.number_input("Enter blood glucose random", min_value=70, max_value=400, value=100)
    blood_urea = st.number_input("Enter blood urea", min_value=10, max_value=200, value=50)
    serum_creatinine = st.number_input("Enter serum creatinine", min_value=0.5, max_value=8.0, value=1.0)
    sodium = st.number_input("Enter sodium", min_value=125, max_value=150, value=135)
    potassium = st.number_input("Enter potassium", min_value=2.5, max_value=7.0, value=4.5)
    haemoglobin = st.number_input("Enter haemoglobin", min_value=5.0, max_value=17.0, value=14.0)
    packed_cell_volume = st.number_input("Enter packed cell volume", min_value=20, max_value=50, value=40)
    white_blood_cell_count = st.number_input("Enter white blood cell count", min_value=4000, max_value=15000, value=8000)
    red_blood_cell_count = st.number_input("Enter red blood cell count", min_value=2.5, max_value=6.0, value=5.0)
    hypertension = st.selectbox("Enter hypertension (0 for no, 1 for yes)", [0, 1])
    diabetes_mellitus = st.selectbox("Enter diabetes mellitus (0 for no, 1 for yes)", [0, 1])
    coronary_artery_disease = st.selectbox("Enter coronary artery disease (0 for no, 1 for yes)", [0, 1])
    appetite = st.selectbox("Enter appetite (0 for poor, 1 for good)", [0, 1])
    peda_edema = st.selectbox("Enter peda edema (0 for no, 1 for yes)", [0, 1])
    aanemia = st.selectbox("Enter aanemia (0 for no, 1 for yes)", [0, 1])

    return [[age, blood_pressure, specific_gravity, albumin, sugar, red_blood_cells, pus_cell,
             pus_cell_clumps, bacteria, blood_glucose_random, blood_urea, serum_creatinine, sodium,
             potassium, haemoglobin, packed_cell_volume, white_blood_cell_count, red_blood_cell_count,
             hypertension, diabetes_mellitus, coronary_artery_disease, appetite, peda_edema, aanemia]]

st.title('Kidney Disease Clinical Trial Eligibility Checker')

user_input = get_user_input()

input_data = pd.DataFrame(user_input, columns=['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
                                               'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                                               'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
                                               'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema', 'aanemia'])

if st.button('Check Eligibility'):
    prediction = model.predict(input_data)[0]
    eligibility = prediction == 0
    st.write(f"Eligibility for clinical trials: {'Eligible' if eligibility else 'Not Eligible'}")
