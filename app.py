import pickle
import joblib
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the models
model = joblib.load('dt_model.joblib')
model1 = joblib.load('lda_model.joblib')
model2 = joblib.load('lr_model.joblib')
model3 = joblib.load('nb_model.joblib')
model4 = joblib.load('rf_model.joblib')
model5 = joblib.load('svm_model.joblib')
model6 = joblib.load('xgb_model.joblib')


@app.route('/')
def index():
    return render_template('main.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Retrieve user input from the form
        gender = request.form.get('gender')
        age = int(request.form['age'])
        cigarettes_per_day = int(request.form['cigarettes'])
        bp_meds = request.form.get('bp_meds')
        prevalent_stroke = request.form.get('stroke')
        prevalent_hypertension = request.form.get('hypertension')
        diabetes = request.form.get('diabetes')
        total_cholesterol = float(request.form['cholesterol'])
        systolic_bp = float(request.form['systolic_bp'])
        diastolic_bp = float(request.form['diastolic_bp'])
        bmi = float(request.form['bmi'])
        heart_rate = float(request.form['heart_rate'])
        glucose = float(request.form['glucose'])

        # Define prepared_data
        prepared_data = [gender, age, cigarettes_per_day, bp_meds, prevalent_stroke, prevalent_hypertension, diabetes, total_cholesterol, systolic_bp, diastolic_bp, bmi, heart_rate, glucose]
        # Convert prepared_data to float
        prepared_data = [float(x) for x in prepared_data]

        # Convert prepared_data to a DataFrame
        prepared_data_df = pd.DataFrame([prepared_data])

        # Generate meta_features
        meta_features = [model.predict(prepared_data_df), model1.predict(prepared_data_df), model2.predict(prepared_data_df),model3.predict(prepared_data_df),
                         model4.predict(prepared_data_df),model5.predict(prepared_data_df),model6.predict(prepared_data_df),]
        # Convert meta_features to a DataFrame
        meta_features_df = pd.DataFrame(meta_features)

        # Concatenate prepared_data_df and meta_features_df
        data_for_prediction = pd.concat([prepared_data_df, meta_features_df], axis=1)

        data_for_prediction = data_for_prediction.iloc[:, :-1]
        
        # Make prediction
        prediction = model.predict(data_for_prediction)[0]

        # Prepare output message based on prediction (example)
        if prediction == 0:
            output = "Low risk of heart disease in 10 years."
        
        else:
            output = "High risk of heart disease in 10 years. Please consult a doctor. In the meantime, consider the following remedies: keep fit, maintain a balanced diet, and regularly check your blood pressure."

        return render_template('result.html', prediction=output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
