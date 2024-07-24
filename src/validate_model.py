import mlflow
import giskard
import numpy as np

# Load the best model (replace 'model' with your actual model path if different)
model_path = "model"
model = mlflow.pyfunc.load_model(model_path)

# Replace with actual test data and labels
test_data = np.array([[0.5, 1.2, 3.4], [1.5, 2.3, 3.4]])
test_labels = np.array([0, 1])

# Validate the model using Giskard
validator = giskard.Validator(model, test_data, test_labels)
validation_report = validator.validate()

# Save the validation report
with open("validation_report.txt", "w") as f:
    f.write(str(validation_report))

print("Validation complete. Report saved to validation_report.txt")
