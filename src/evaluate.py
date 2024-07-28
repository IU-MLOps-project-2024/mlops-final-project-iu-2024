import argparse

import mlflow
import sklearn
import sklearn.metrics
from model import load_features

def evaluate_model(model, X_test, y_test, metrics_fns):
    predictions = model.predict(X_test)
    metrics = {}
    for metric_name, metric_fn in metrics_fns.items():
        metrics[metric_name] = metric_fn(y_test, predictions, average="weighted")
    
    return metrics

def evaluate(data_version, model_name, model_alias = "champion") -> None:
    model_uri = f"models:/{model_name}@{model_alias}"
    model = mlflow.sklearn.load_model(model_uri=model_uri)

    experiment_name = model_name + "_eval"
    print(f"Evaluating model {model_name}@{model_alias} on data sample {data_version}")

    X_test, y_test = load_features(name="features_target", version=data_version)

    metrics_fns = {
        "Recall": sklearn.metrics.recall_score,
        "Presicion": sklearn.metrics.precision_score,
        "F1": sklearn.metrics.f1_score
    }

    test_metrics = evaluate_model(model, X_test, y_test, metrics_fns)
    
    for metric_name, value in test_metrics.items():
        print(f'{metric_name}: {value:.5f}')
    
    experiment_name = model_name + "_evaluate" 

    try:
        # Create a new MLflow Experiment
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id

    with mlflow.start_run(run_name=f"{model_name}_{model_alias}_{data_version}_evaluate",
                          experiment_id=experiment_id):
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("model_alias", model_alias)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_alias", model_alias)
        mlflow.log_param("data_version", data_version)

        for metric_name, value in test_metrics.items():
            mlflow.log_metric(metric_name, value)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="DecisionTreeClassifier")
    parser.add_argument("--model-alias", type=str, default="champion")
    parser.add_argument("--data-version", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.data_version, args.model_name, args.model_alias)


if __name__ == "__main__":
    main()
