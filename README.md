# mlops-final-project-iu-2024
The final project of the MLOps course at Innopolis University 2024

![Test code workflow](https://github.com/IU-MLOps-project-2024/mlops-final-project-iu-2024/actions/workflows/test-code.yaml/badge.svg)
<!-- [Validate model workflow](https://github.com/IU-MLOps-project-2024/mlops-final-project-iu-2024/actions/workflows/validate-model.yaml/badge.svg) -->

## How to deploy

1. Run docker container
docker run -d -p 5152:8080 --name category_container datapaf/category_model

2. Run Flask API
python3 api/app.py

3. Run Gradio UI
python3 src/app.py
