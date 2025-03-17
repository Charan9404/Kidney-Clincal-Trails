import joblib
import pandas as pd

model = joblib.load('kidney_disease_model.pkl')


user_input = [25, 120, 1.025, 0, 0, 1, 1, 0, 0, 100, 20, 1.0, 140, 4.0, 15, 42, 8000, 4.8, 0, 0, 0, 1, 0, 0]

input_data = pd.DataFrame([user_input], columns=['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
                                               'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                                               'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
                                               'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema', 'aanemia'])
prediction = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0]
eligibility = prediction == 0
print(prediction)
print(prob[0])
print(f"Eligibility for clinical trials: {'Eligible' if eligibility else 'Not Eligible'}")

#[25, 120, 1.025, 0, 0, 1, 1, 0, 0, 100, 20, 1.0, 140, 4.0, 15, 42, 8000, 4.8, 0, 0, 0, 1, 0, 0]
#[26, 89, 1.019, 0, 2, 0, 0, 1, 0, 187, 157, 0.72, 142, 6.17, 16.18, 46, 7513, 4.55, 0, 0, 0, 0, 1, 0]
#[51, 101, 1.020, 2, 3, 1, 1, 1, 0, 160, 39, 3.4, 140, 4.4, 10.7, 34, 10260, 3.59, 1, 0, 0, 0, 0, 1]
#[34, 96, 1.022, 1, 5, 0, 0, 1, 1, 234, 115, 5.89, 131, 6.9, 16.69, 40, 6195, 4.88, 0, 0, 0, 0, 1, 1]