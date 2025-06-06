# app.py — Employee Attrition Prediction App

import streamlit as st
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load and prepare data
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Encode categorical variables
le_dict = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Train-test split and SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train the model
model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train_res, y_train_res)

# Streamlit page setup
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("🔮 Employee Attrition Prediction App")
st.write("Fill in the employee details to predict if they're at risk of leaving.")

# User input collection
def user_input_features():
    input_data = {}
    for col in X.columns:
        if col in le_dict:
            options = list(le_dict[col].classes_)
            input_val = st.selectbox(f"{col}", options)
            input_data[col] = le_dict[col].transform([input_val])[0]
        else:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            col_mean = float(df[col].mean())
            if col_min == col_max:
                st.write(f"⚠️ Skipping '{col}' (constant value: {col_min})")
                input_data[col] = col_min
            else:
                input_data[col] = st.slider(f"{col}", col_min, col_max, col_mean)
    return pd.DataFrame([input_data])

input_df = user_input_features()

# Predict
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# Show prediction result
st.subheader(" Prediction Result")
if prediction == 1:
    st.error(f" This employee is likely to leave. (Confidence: {probability * 100:.2f}%)")
else:
    st.success(f" This employee is likely to stay. (Confidence: {(1 - probability) * 100:.2f}%)")

# SHAP explainability
st.subheader(" SHAP Feature Explanation")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(input_df)

shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(bbox_inches="tight")
