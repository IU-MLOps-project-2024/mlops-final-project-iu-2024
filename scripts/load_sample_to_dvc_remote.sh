#!/bin/bash

cd ~/Desktop/mlops-final-project-iu-2024/

# Add the data file to DVC
dvc add data/samples/sample.csv
if [ $? -ne 0 ]; then
  echo "Failed to add data file to DVC"
  exit 1
fi

# Add the DVC remote and set it as default
dvc remote add --force -d localstore ~/Desktop/mlops-final-project-iu-2024/datastore/
if [ $? -ne 0 ]; then
  echo "Failed to add or set DVC remote"
  exit 1
fi

# Push the data to the DVC remote
dvc push
if [ $? -ne 0 ]; then
  echo "Failed to push data to DVC remote"
  exit 1
fi
