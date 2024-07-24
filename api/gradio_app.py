import gradio as gr
import requests
import json

# Define the function to get predictions from the Flask API
def get_prediction(feature1, feature2, feature3):
    url = "http://localhost:5000/predict"  # Update this if using a cloud API
    input_data = {"features": [feature1, feature2, feature3]}
    response = requests.post(url, json=input_data)
    
    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        return f"Error: {response.json()['error']}"

# Define the Gradio interface
def predict_interface(feature1, feature2, feature3):
    prediction = get_prediction(feature1, feature2, feature3)
    return prediction

iface = gr.Interface(
    fn=predict_interface,
    inputs=[
        gr.inputs.Number(label="Feature 1"),
        gr.inputs.Number(label="Feature 2"),
        gr.inputs.Number(label="Feature 3")
    ],
    outputs="text",
    title="ML Model Prediction",
    description="Enter the feature values to get the model prediction."
)

# Launch the Gradio app
iface.launch(share=True)  # Set share=True to deploy to Gradio Cloud
