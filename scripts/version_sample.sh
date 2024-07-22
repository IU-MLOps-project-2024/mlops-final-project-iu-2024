#!/bin/bash

# Check if version argument is provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <data_path> <version>"
  exit 1
fi

DATA_PATH=$1
NEW_VERSION=$2

# Add the data to DVC
dvc add "$DATA_PATH"
if [ $? -ne 0 ]; then
  echo "Failed to add data to DVC"
  exit 1
fi

# Stage the changes for Git
git add .
if [ $? -ne 0 ]; then
  echo "Failed to add changes to Git"
  exit 1
fi

# Commit the changes to Git with the version
git commit -m "Version $NEW_VERSION of sample data"
if [ $? -ne 0 ]; then
  echo "Failed to commit changes to Git"
  exit 1
fi

# Push the changes to DVC remote
dvc push
if [ $? -ne 0 ]; then
  echo "Failed to push data to DVC remote"
  exit 1
fi
