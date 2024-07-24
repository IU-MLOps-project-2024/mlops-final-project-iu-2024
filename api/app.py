from flask import Flask, request, jsonify
import mlflow.pyfunc
import json
import os

app = Flask(__name__)

# Load the MLflow model
model_path = "model"
model = mlflow.pyfunc.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data["features"]
        prediction = model.predict([features])
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/info', methods=['GET'])
def info():
    try:
        model_metadata = model.metadata.to_dict()
        return jsonify(model_metadata)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
