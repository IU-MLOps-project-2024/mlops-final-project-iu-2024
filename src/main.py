"""Test"""
import hydra
from model import train, load_features, log_metadata


def run(args):
    """Run function"""
    cfg = args

    train_data_version = cfg.train_data_version

    X_train, y_train = load_features(name = "features_target", version=train_data_version)

    test_data_version = cfg.test_data_version

    X_test, y_test = load_features(name = "features_target", version=test_data_version)

    gs = train(X_train, y_train, cfg=cfg)

    log_metadata(cfg, gs, X_train, y_train, X_test, y_test)


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg=None):
    """Main function"""
    run(cfg)


if __name__=="__main__":
    main()
