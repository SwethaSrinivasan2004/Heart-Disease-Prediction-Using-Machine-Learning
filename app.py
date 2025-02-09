from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained KNN model and scaler using pickle
with open('knn.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


@app.route('/')
def home():
    return render_template('index.html')  # Front-end form


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse features from the form
        features = [float(x) for x in request.form.values()]

        # Scale the features
        features_scaled = scaler.transform([features])

        # Make prediction
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)[:, 1]

        # Return result to the front-end
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': round(prediction_proba[0], 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)