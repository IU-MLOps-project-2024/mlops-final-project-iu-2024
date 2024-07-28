from collections import defaultdict
import os
import random

import numpy as np
import sklearn.metrics
import torch
import hydra
from omegaconf import DictConfig
import sklearn
from model import load_features, train

def evaluate_model(gs, X_test, y_test, metrics_fns):
    predictions = gs.best_estimator_.predict(X_test)
    metrics = {}
    for metric_name, metric_fn in metrics_fns.items():
        metrics[metric_name] = metric_fn(y_test, predictions, average="weighted")
    
    return metrics

@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg: DictConfig=None):
    results_path = os.path.join(cfg.paths.root_path, "results")

    test_metrics = defaultdict(list)
    metrics_fns = {
        "Recall": sklearn.metrics.recall_score,
        "Presicion": sklearn.metrics.precision_score,
        "F1": sklearn.metrics.f1_score
    }

    print(f'Begin check for {cfg.model.model_name}')
    for seed in cfg.reproducibility.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        cfg.data.random_state = seed

        X_train, y_train = load_features(name="features_target", version=cfg.train_data_version)
        X_test, y_test = load_features(name="features_target", version=cfg.test_data_version)

        gs = train(X_train, y_train, cfg=cfg)

        eval = evaluate_model(gs, X_test, y_test, metrics_fns)
        for metric_name, value in eval.items():
            test_metrics[metric_name].append(value)

    f = os.path.join(results_path, f"reproducibility_{cfg.model.model_name}.txt")
    with open(str(f), "w") as f:
        f.write(f"Test metrics for model {cfg.model.model_name}:\n")
        for metric_name, metrics in test_metrics.items():
            f.write(f"  {metric_name}: {metrics}\n")
            avg_metric = np.mean(metrics)
            var_metric = np.var(metrics)
            f.write(f"  Average of {metric_name}: {avg_metric}\n")
            f.write(f"  Variance of {metric_name}: {var_metric}\n\n")

if __name__ == "__main__":
    main()
