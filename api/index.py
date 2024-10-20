from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


df = pd.read_excel('DATA_ALMODEL.xlsx') 
categorical_cols = ['COUNTRY', 'FORMATION', 'FIELD ', 'PROD PATH']
le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Load the trained model
with open('pipe_svc.pkl', 'rb') as file:
    model = pickle.load(file)
    
X = df.drop('PROD PATH', axis=1)
y = df['PROD PATH']

# Load the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = pd.DataFrame([data])

    input_data = input_data.rename(columns={
        'CUM WATER PROD, RB': 'CUM WATER PROD, RB',
        'DAILY WATER PROD, RB/D': 'DAILY WATER PROD , RB/D',
        'FIELD ': 'FIELD'
    })

    required_columns = X.columns
    for col in required_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[required_columns]
    input_scaled = scaler.transform(input_data)

    prediction_proba = model.predict_proba(input_scaled)
    actual_output = None
    matching_row = X[X.eq(input_data.iloc[0]).all(axis=1)]
    if not matching_row.empty:
        actual_output = y.iloc[matching_row.index[0]]

    result = {
        'predicted_probabilities': prediction_proba.tolist(),
        'actual_output': int(actual_output) if actual_output is not None else None
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
