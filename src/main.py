"""Test"""
import hydra
import sklearn.datasets
import sklearn.model_selection
from model import train, load_features, log_metadata

import pandas as pd
import sklearn
import numpy as np


def run(args):
    """Run function"""
    cfg = args

    train_version = cfg.train_data_version

    X_train, y_train = load_features(name = "features_target", version=train_version)

    test_version = cfg.test_data_version

    X_test, y_test = load_features(name = "features_target", version=test_version)

    gs = train(X_train, y_train, cfg=cfg)

    log_metadata(cfg, gs, X_train, y_train, X_test, y_test)


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg=None):
    """Main function"""
    run(cfg)


if __name__=="__main__":
    main()
