# prompt: make a flask api for the model with an index get api

from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return '<h1>Welcome to the Solar Cell Performance Prediction API!</h1>'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    QW = data['QW']
    S_Comp = data['S_Comp']
    NGD = data['NGD']
    WGD = data['WGD']

    model = joblib.load('model.joblib')
    predicted_performance = model.predict([[QW, S_Comp, NGD, WGD]])[0]

    Isc, Jsc, Voc, FF, Efficiency = predicted_performance

    return jsonify({
        'Isc': Isc,
        'Jsc': Jsc,
        'Voc': Voc,
        'FF': FF,
        'Efficiency': Efficiency
    })

if __name__ == '__main__':
    app.run(debug=True)
