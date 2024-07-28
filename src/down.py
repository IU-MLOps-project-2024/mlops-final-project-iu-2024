import mlflow
from mlflow.tracking import MlflowClient

def download_model_artifacts(model_name, alias, destination_path):
    """
    Downloads the model artifacts for the specified model and alias from MLflow.

    Args:
        model_name (str): The name of the registered model.
        alias (str): The alias of the model version (e.g., 'Production', 'Staging').
        destination_path (str): The local directory where the artifacts will be downloaded.

    Returns:
        None
    """
    # Initialize MLflow client
    client = MlflowClient()
    
    # Get the latest version of the model for the given alias
    model_version = client.get_model_version_by_alias(model_name, alias)
    
    # Get the run_id and artifact_path from the model version
    artifact_path = model_version.source
    
    # Download the artifacts to the destination path
    mlflow.artifacts.download_artifacts(artifact_uri=artifact_path, dst_path=destination_path)

    print(f"Model artifacts for {model_name} (alias: {alias}) downloaded to {destination_path}")

# Example usage
models = ['Transformer', 'MLP', 'LogisticRegression', 'DecisionTreeClassifier']
aliases = ['challenger1', 'challenger2', 'challenger3', 'challenger4', 'challenger5', 'champion']
for model in models:
    for alias in aliases:
        try:
            destination_path = f'./models/{model}/{alias}/'
            download_model_artifacts(model, alias, destination_path)
        except:
            print(f"No {alias} for {model}")
