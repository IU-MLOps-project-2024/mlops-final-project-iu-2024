#!/bin/bash

# Check if all required arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <data_path> <tag_name> <branch>"
    exit 1
fi

data_path="$1"
tag_name="$2"
branch="$3"

# Step 1: Read and Validate Data
if [ ! -s "$data_path" ]; then
    echo "Error: $data_path is empty or doesn't exist."
    exit 1
fi

# Step 2: Version the data sample (only if valid)
# Example: Assume validation is successful (modify as per your actual validation)
data_valid=true

# Step 3: Version the data using DVC
if [ "$data_valid" = true ]; then
    # Version data with DVC
    dvc add "$data_path"
    dvc commit -m "Versioning sample data"
fi

# Step 4: Git operations
# Add changes to Git
git add "$data_path.dvc"   # Assuming DVC creates a .dvc file for versioning
git add "$data_path"       # Add original data file if needed

# Commit changes
git commit -m "Update sample data"

# Push changes to specified branch on GitHub
git push origin "$branch"

# Step 5: Tag the commit
commit_hash=$(git rev-parse HEAD)

git tag -a "$tag_name" "$commit_hash" -m "Version $tag_name"
git push origin "$tag_name"

# Step 6: Push to DVC (assuming DVC remote is set up)
if [ "$data_valid" = true ]; then
    dvc push
fi

echo "Script execution completed."