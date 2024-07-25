#!/bin/bash

# List of versions to test
versions=(0 1 2 3 4)

# Hostname and port for the prediction service
hostname="localhost"
port=5152
random_state=42

# Run data_prepare.py 5 times
for version in "${versions[@]}"; do
  echo "Running data_prepare.py for version ${version}..."
  python3 ~/Desktop/mlops-final-project-iu-2024/pipelines/data_prepare.py
  if [ $? -ne 0 ]; then
    echo "data_prepare.py failed for version ${version}"
    exit 1
  fi
done

echo "All data preparation runs completed successfully."

# Loop through each version and test the prediction service
for version in "${versions[@]}"; do
  echo "Testing prediction service with example_version=${version}..."
  mlflow run . --env-manager local -e predict -P example_version=${version} -P hostname=${hostname} -P port=${port} -P random_state=${random_state}
  if [ $? -ne 0 ]; then
    echo "Prediction test failed for version ${version}"
    exit 1
  fi
done

echo "All prediction tests passed successfully."