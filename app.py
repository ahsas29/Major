from flask import Flask, request, jsonify
from joblib import dump, load
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

model = load('model.joblib')
scaler = load('scaler.joblib')
model1 = load('model1.joblib')
scaler2 = load('scaler2.joblib')

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Welcome to the solar cell efficiency prediction API."})

@app.route('/predict/g-dana', methods=['POST'])
def predict_g_dana():
    data = request.get_json()
    features = np.array([[data['QW'], data['S_Comp'], data['NGD'], data['WGD']]])
    features_scaled = scaler.transform(features)
    efficiency = model.predict(features_scaled)[0]
    return jsonify({"efficiency": efficiency.tolist()})

@app.route('/predict/t-dana', methods=['POST'])
def predict_t_dana():
    data = request.get_json()
    features = np.array([[data['QW'], data['S_Comp'], data['NGD'], data['WGD']]])
    features_scaled = scaler2.transform(features)
    efficiency = model1.predict(features_scaled)[0]
    return jsonify({"efficiency": efficiency.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
