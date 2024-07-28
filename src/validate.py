import giskard
import mlflow
import os
import numpy as np
from model import load_features
from sklearn.metrics import f1_score
from hydra import compose, initialize
import argparse
import pandas as pd


def create_giskard_dataset(data_version, cfg):
    target_col = cfg.datasets.target_col
    X_test, y_test = load_features(name="features_target", version=data_version)
    dataset_name = f"sample_{data_version}_giskard"

    df = pd.DataFrame(X_test)
    df[target_col] = y_test

    giskard_dataset = giskard.Dataset(
        df=df,
        target=target_col,
        name=dataset_name
    )

    return giskard_dataset, df


def load_model(model_name, model_version, feature_cols, cfg):
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri=model_uri)

    target_col = cfg.datasets.target_col

    def predict_fn(X):
        y_pred = model.predict(X).flatten()
        return y_pred

    giskard_model = giskard.Model(
        model=predict_fn,
        target=target_col,
        model_type="classification",
        name=f'{model_name}_v{model_version}',
        classification_labels=list(range(20))
    )

    return giskard_model


def run_giskard_scan(giskard_model, giskard_dataset, model_name, model_version, data_version, cfg):
    root_dir = cfg.paths.root_path
    scan_path = os.path.join(root_dir, "reports",
                             f"test_suite_{model_name}_v{model_version}_sample_{data_version}.html")
    scan = giskard.scan(giskard_model, giskard_dataset)
    scan.to_html(scan_path)

    return scan


def run_giskard_tests(giskard_model, giskard_dataset, model_name, model_version,
                      data_version, f1_threshold=0.2):
    suite_name = f"test_suite_{model_name}_v{model_version}_sample_{data_version}"
    test_suite = giskard.Suite(name=suite_name)

    def f1_test_fn(model, data):
        y_true = data.df[data.target].values.astype(np.float32).reshape(-1, 1)
        y_pred = model.predict(data).raw.reshape(-1, 1)
        f1 = f1_score(y_true, y_pred)
        return giskard.TestResult(
            metric_name="F1",
            metric=f1,
            passed=f1 > f1_threshold,
        )

    test_suite.add_test(f1_test_fn, model=giskard_model, data=giskard_dataset,
                        display_name=f"F1 Test (Threshold {f1_threshold})", test_id="f1_test")

    test_results = test_suite.run()
    return test_results


def validate_model(model_name, model_alias, model_version,
                   data_version, f1_threshold, giskard_dataset, cfg):
    if model_version is None and (model_alias is None or model_alias == ""):
        raise ValueError("Either model_alias or model_version should be provided")

    client = mlflow.MlflowClient()
    if model_alias is not None and model_alias != "":
        print(client.get_latest_versions(model_name))
        model_version = client.get_model_version_by_alias(model_name, model_alias).version

    feature_cols = giskard_dataset.df.columns.tolist()
    target_col = cfg.datasets.target_col
    feature_cols.remove(target_col)

    giskard_model = load_model(model_name, model_version, feature_cols, cfg)
    scan = run_giskard_scan(giskard_model, giskard_dataset, model_name, model_version, data_version, cfg)
    test_results = run_giskard_tests(giskard_model, giskard_dataset, model_name, model_version,
                                     data_version, f1_threshold)
    return scan, test_results, model_version


def validate_all_models(model_name, data_version, f1_threshold, giskard_dataset, cfg):
    client = mlflow.MlflowClient()
    num_challengers = cfg.model.num_challengers
    challenger_aliases = [f"challenger{i+1}" for i in range(num_challengers)]

    best_model = None
    min_issues = float("inf")

    for challenger_alias in challenger_aliases:
        try:
            scan, test_results, model_version = validate_model(model_name, challenger_alias, None,
                                                               data_version, f1_threshold, giskard_dataset, cfg)
            if test_results.passed:
                num_issues = len(scan.issues)
                if num_issues < min_issues:
                    best_model = (challenger_alias, model_version)
                    min_issues = num_issues
        except Exception as e:
            print(f"Error validating model {challenger_alias}: {e}")

    if best_model is not None:
        print(f"Promoting {best_model} to champion")
        client.transition_model_version_stage(name=model_name,
                                              version=best_model[1],
                                              stage="Production")

        client.delete_registered_model_alias(name=model_name, alias="champion")
        client.set_registered_model_alias(name=model_name, alias="champion", version=best_model[1])

        return best_model
    else:
        print("No challenger model passed the tests")
        return None


def main():
    with initialize(config_path="../configs", job_name="validate", version_base=None):
        cfg = compose(config_name="main")
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-name", type=str)
        parser.add_argument("--model-alias", type=str, default=None)
        parser.add_argument("--model-version", type=int, default=None)
        parser.add_argument("--data-version", type=str, default="5", required=True)
        parser.add_argument("--f1-threshold", type=float, default=0.2)
        parser.add_argument("--validate-all", action="store_true")
        args = parser.parse_args()

        giskard_dataset, df = create_giskard_dataset(args.data_version, cfg)

        if args.validate_all:
            print("Validating all challenger models")
            best_model = validate_all_models(args.model_name, args.data_version, args.f1_threshold, giskard_dataset,
                                             cfg)
            if best_model is not None:
                print(f"Best model: {best_model}")
        else:
            scan, test_results, model_version = validate_model(args.model_name, args.model_alias, args.model_version,
                                                               args.data_version, args.f1_threshold, giskard_dataset,
                                                               cfg)
            print(f"Model version: {model_version}")
            print(f"Scan issues: {len(scan.issues)}")
            print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()