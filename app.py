# prompt: make a flask api for the model with an index get api

from flask import Flask, request, jsonify
from joblib import dump, load
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

model = load('model.joblib')
scaler = load('scaler.joblib')

def predict_efficiency(QW, S_Comp, NGD, WGD):
  """
  Predicts the efficiency of a solar cell based on its features.

  Args:
    QW: The quantum well width in nanometers.
    S_Comp: The composition of the quantum well.
    NGD: The number of growth directions.
    WGD: The width of the growth direction in nanometers.

  Returns:
    The predicted efficiency of the solar cell.
  """

  # Normalize the features
  features = np.array([[QW, S_Comp, NGD, WGD]])
  features_scaled = scaler.transform(features)

  # Predict the efficiency
  efficiency = model.predict(features_scaled)[0]

  return efficiency

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
  return jsonify({"message": "Welcome to the solar cell efficiency prediction API."})

@app.route('/predict', methods=['POST'])
def predict():
  data = request.get_json()
  features = np.array([[data['QW'], data['S_Comp'], data['NGD'], data['WGD']]])
  features_scaled = scaler.transform(features)
  efficiency = model.predict(features_scaled)[0]
  return jsonify({"efficiency": efficiency.tolist()})

if __name__ == '__main__':
  app.run(debug=True)