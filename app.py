import pandas as pd
import numpy as np

data = pd.read_csv("kidney_disease.csv")
data.drop('id',axis=1, inplace=True)

data.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'aanemia', 'class']

data['packed_cell_volume'] = pd.to_numeric(data['packed_cell_volume'], errors='coerce')
data['white_blood_cell_count'] = pd.to_numeric(data['white_blood_cell_count'], errors='coerce')
data['red_blood_cell_count'] = pd.to_numeric(data['red_blood_cell_count'], errors='coerce')

cat_cols = [col for col in data.columns if data[col].dtype == 'object']
num_cols = [col for col in data.columns if data[col].dtype != 'object']

data['diabetes_mellitus'].replace(to_replace = {'\tno':'no', '\tyes': 'yes', ' yes':'yes'}, inplace=True)
data['coronary_artery_disease'] = data['coronary_artery_disease'].replace(to_replace = '\tno', value = 'no')

data['class'] = data['class'].replace(to_replace={'ckd\t':'ckd', 'notckd': 'not ckd'})
data['class'] = data['class'].map({'ckd':0, 'not ckd': 1})
data['class'] = pd.to_numeric(data['class'], errors = 'coerce')

def random_sampling(feature):
    random_sample = data[feature].dropna().sample(data[feature].isna().sum())
    random_sample.index = data[data[feature].isnull()].index
    data.loc[data[feature].isnull(), feature] = random_sample

def impute_mode(feature):
    mode = data[feature].mode()[0]
    data[feature] = data[feature].fillna(mode)

for col in num_cols:
    random_sampling(col)

random_sampling('red_blood_cells')
random_sampling('pus_cell')
for col in cat_cols:
    impute_mode(col)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

X = data.drop('class', axis = 1)
Y = data['class']

from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test =  train_test_split(X,Y, test_size = 0.2, random_state = 40)
print(Y_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
rand_clf = RandomForestClassifier(criterion = "gini", max_depth = 10, max_features="sqrt", min_samples_leaf= 1, min_samples_split= 7, n_estimators = 400)
rand_clf.fit(X_train, Y_train)
rand_clf_acc = accuracy_score(Y_test, rand_clf.predict(X_test))
print(f"Training Accuracy of Random Forest is {accuracy_score(Y_train, rand_clf.predict(X_train))}")
print(f"Testing Accuracy of Random Forest is {accuracy_score(Y_test, rand_clf.predict(X_test))}")
print(f"Confusion Matrix of Random Forest is \n {confusion_matrix(Y_test, rand_clf.predict(X_test))}\n")
print(f"Classification Report of Random Forest is \n{classification_report(Y_test, rand_clf.predict(X_test))}")

import joblib
joblib.dump(rand_clf, 'kidney_disease_model.pkl')