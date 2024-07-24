#!/bin/bash

# List of versions to test
versions=("v1" "v2" "v3" "v4" "v5")

# Hostname and port for the prediction service
hostname="localhost"
port=5151
random_state=42

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
