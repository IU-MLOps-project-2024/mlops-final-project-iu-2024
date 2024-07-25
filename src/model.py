"""Script for running models"""
import importlib
import sys
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from zenml.client import Client
import pandas as pd
import mlflow
import mlflow.sklearn
import torch
import skorch
import giskard
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mlp import MLP

def load_features(name, version, size = 1):
    """Load features"""
    client = Client()
    l = client.list_artifact_versions(name = name, sort_by="version").items
    l.reverse()

    df = l[version].load()
    df = df.sample(frac = size, random_state = 88)

    # print("size of df is ", df.shape)
    # print("df columns: ", df.columns)

    X = df[df.columns[:-1]].to_numpy(dtype=np.float32)
    y = df[df.columns[-1]].to_numpy()

    # print("shapes of X,y = ", X.shape, y.shape)

    return X, y


def log_metadata(cfg, gs, X_train, y_train, X_test, y_test):
    """Log metadata"""
    cv_results = pd.DataFrame(gs.cv_results_).filter(regex=r'std_|mean_|param_').sort_index(axis=1)
    best_metrics_values = [result[1][gs.best_index_] for result in gs.cv_results_.items()]
    best_metrics_keys = [metric for metric in gs.cv_results_]
    best_metrics_dict = {k:v for k,v in zip(best_metrics_keys, best_metrics_values) if 'mean' in k or 'std' in k}

    df_train = pd.DataFrame(np.concatenate([X_train, y_train[:, np.newaxis]], axis=1))
    df_test = pd.DataFrame(np.concatenate([X_test, y_test[:, np.newaxis]], axis=1))

    experiment_name = cfg.model.model_name + "_" + cfg.experiment_name

    try:
        # Create a new MLflow Experiment
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id

    print("experiment-id : ", experiment_id)

    run_name = "_".join([cfg.run_name, cfg.model.model_name, cfg.model.evaluation_metric])
    print("run name: ", run_name)

    if mlflow.active_run():
        mlflow.end_run()

    # Fake run
    with mlflow.start_run():
        pass

    # Parent run
    with mlflow.start_run(run_name = run_name, experiment_id = experiment_id):
        df_train_dataset = mlflow.data.pandas_dataset.from_pandas(df = df_train)
        df_test_dataset = mlflow.data.pandas_dataset.from_pandas(df = df_test)
        mlflow.log_input(df_train_dataset, "training")
        mlflow.log_input(df_test_dataset, "testing")

        # Log the hyperparameters
        mlflow.log_params(gs.best_params_)

        # Log the performance metrics
        mlflow.log_metrics(best_metrics_dict)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag(cfg.model.tag_key, cfg.model.tag_value)

        # Infer the model signature
        signature = mlflow.models.infer_signature(X_train[0], gs.predict(X_train)[0])

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model = gs.best_estimator_,
            artifact_path = cfg.model.artifact_path,
            signature = signature,
            input_example = X_train[0],
            registered_model_name = cfg.model.model_name,
            pyfunc_predict_fn = cfg.model.pyfunc_predict_fn
        )

        model_uri = model_info.model_uri
        loaded_model = mlflow.sklearn.load_model(model_uri=model_uri)

        predictions = loaded_model.predict(X_test) # type: ignore
        eval_data = pd.DataFrame(y_test)
        eval_data.columns = ["label"]
        eval_data["predictions"] = predictions

        results = mlflow.evaluate(
            data=eval_data,
            model_type="classifier",
            targets="label",
            predictions="predictions",
            evaluators=["default"]
        )

        print(f"metrics:\n{results.metrics}")

        client = mlflow.client.MlflowClient()
        client.set_model_version_tag(name = cfg.model.model_name, version=model_info.registered_model_version, key="source", value="best_Grid_search_model")

        for index, result in cv_results.iterrows():
            child_run_name = "_".join(['child', run_name, str(index)])
            with mlflow.start_run(run_name = child_run_name, experiment_id= experiment_id, nested=True):
                ps = result.filter(regex='param_').to_dict()
                ms = result.filter(regex='mean_').to_dict()
                stds = result.filter(regex='std_').to_dict()

                # Remove param_ from the beginning of the keys
                ps = {k.replace("param_",""):v for (k,v) in ps.items()}
                ps = {k.replace("module__",""):v for (k,v) in ps.items()}

                mlflow.log_params(ps)
                mlflow.log_metrics(ms)
                mlflow.log_metrics(stds)

                # We will create the estimator at runtime
                module_name = cfg.model.module_name

                if module_name == "torch":
                    num_layers = int(ps['num_layers'])
                    hidden_size = int(ps['hidden_size'])
                    lr = ps['lr']
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    if device == "cuda" and torch.cuda.get_device_capability() == (8, 9):
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                    estimator = skorch.NeuralNetClassifier(
                        MLP(num_layers=num_layers, hidden_size=hidden_size),
                        max_epochs=50,
                        criterion=torch.nn.CrossEntropyLoss,
                        device=device,
                        iterator_train__shuffle=True,
                        optimizer=torch.optim.AdamW,
                        lr=lr
                    )
                elif module_name.startswith("sklearn"):
                    class_name  = cfg.model.class_name
                    class_instance = getattr(importlib.import_module(module_name), class_name)
                    estimator = class_instance(**ps)
                else:
                    raise ValueError("This library is not supported")

                estimator.fit(X_train, y_train)

                signature = mlflow.models.infer_signature(X_train[0], estimator.predict(X_train)[0])

                model_info = mlflow.sklearn.log_model(
                    sk_model = estimator,
                    artifact_path = cfg.model.artifact_path,
                    signature = signature,
                    input_example = X_train[0],
                    registered_model_name = cfg.model.model_name,
                    pyfunc_predict_fn = cfg.model.pyfunc_predict_fn
                )

                model_uri = model_info.model_uri
                loaded_model = mlflow.sklearn.load_model(model_uri=model_uri)

                predictions = loaded_model.predict(X_test)
                eval_data = pd.DataFrame(y_test)
                eval_data.columns = ["label"]
                eval_data["predictions"] = predictions

                results = mlflow.evaluate(
                    data=eval_data,
                    model_type="classifier",
                    targets="label",
                    predictions="predictions",
                    evaluators=["default"]
                )

                print(f"metrics:\n{results.metrics}")


def train(X_train, y_train, cfg):
    """Train model"""
    # Define the model hyperparameters
    params = cfg.model.params

    # Train the model
    module_name = cfg.model.module_name

    if module_name == "torch":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and torch.cuda.get_device_capability() == (8, 9):
            # Some optimization for my GPU
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        estimator = skorch.NeuralNetClassifier(
            MLP,
            max_epochs=50,
            criterion=torch.nn.CrossEntropyLoss,
            device=device,
            iterator_train__shuffle=True,
            optimizer=torch.optim.AdamW
        )
    elif module_name.startswith("sklearn"):
        class_name  = cfg.model.class_name
        class_instance = getattr(importlib.import_module(module_name), class_name)
        estimator = class_instance(**params)
    else:
        raise ValueError("This library is not supported")

    cv = StratifiedKFold(n_splits=cfg.model.folds, random_state=cfg.random_state, shuffle=True)

    param_grid = dict(params)

    scoring = list(cfg.model.metrics.values())

    evaluation_metric = cfg.model.evaluation_metric

    gs = GridSearchCV(
        estimator = estimator,
        param_grid = param_grid,
        scoring = scoring,
        n_jobs = cfg.cv_n_jobs,
        refit = evaluation_metric,
        cv = cv,
        verbose = 1,
        return_train_score = True
    )

    gs.fit(X_train, y_train)

    return gs


def retrieve_model_with_alias(model_name, model_alias = "champion") -> mlflow.pyfunc.PyFuncModel:
    """Retrieve model with alias"""
    best_model:mlflow.pyfunc.PyFuncModel = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{model_alias}")

    return best_model

def retrieve_model_with_version(model_name, model_version = "v1") -> mlflow.pyfunc.PyFuncModel:
    """Retrieve model with version"""
    best_model:mlflow.pyfunc.PyFuncModel = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

    return best_model
