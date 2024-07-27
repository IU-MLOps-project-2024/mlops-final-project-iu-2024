# api/app.py

from flask import Flask, request, jsonify, abort, make_response

import mlflow
import mlflow.pyfunc
import os

import numpy as np
import json
import requests

BASE_PATH = '/home/datapaf/Desktop/mlops-final-project-iu-2024'

model = mlflow.pyfunc.load_model(os.path.join(BASE_PATH, "api", "model_dir"))

app = Flask(__name__)

@app.route("/info", methods = ["GET"])
def info():
	response = make_response(str(model.metadata), 200)
	response.content_type = "text/plain"
	return response

@app.route("/", methods = ["GET"])
def home():
	msg = """
	Welcome to our ML service to predict commodities category\n\n

	This API has two main endpoints:\n
	1. /info: to get info about the deployed model.\n
	2. /predict: to send predict requests to our deployed model.\n

	"""

	response = make_response(msg, 200)
	response.content_type = "text/plain"
	return response

# /predict endpoint
@app.route("/predict", methods = ["POST"])
def predict():
    try:
        data = request.json
        features = data["features"]
        input_data = {
			"inputs": [features]
		}
        # prediction = model.predict(np.array([features], dtype=np.float32))
        response = requests.post(
			url=f"http://localhost:5152/invocations",
			data=json.dumps(input_data),
			headers={"Content-Type": "application/json"},
		)
        # print(response.json())
        return jsonify(response.json())
        # return 
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# This will run a local server to accept requests to the API.
if __name__ == "__main__":
    # port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=5001)