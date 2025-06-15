import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import mlflow
import optuna
import pandas as pd
from tqdm.contrib.logging import logging_redirect_tqdm

from src.config import DEFAULT_RANKER_HYPERPARAMETERS_PIPELINE_CONFIG
from src.experiment_tracking import log_config, log_model_params, setup_mlflow
from src.feature_extraction import load_optimized_raw_data
from src.input_preprocessing import (
    LightGBMDataResult,
    get_path_to_lightgbm_data,
)
from src.metrics import get_mapping_from_labels, mean_average_precision_at_k
from src.ranker import prepare_features_for_ranker

logger = logging.getLogger(__name__)


def get_default_config_ranker_with_hyperparameter_tuning():
    return DEFAULT_RANKER_HYPERPARAMETERS_PIPELINE_CONFIG


def evaluate_mapk_train(model, X, ids, labels):
    """Evaluate MAP@K. Assumes only one group per customer (one week_num)."""
    if ids.week_num.nunique() > 1:
        raise ValueError("Assumes only one group per customer (one week_num)")

    scores = model.predict(X)
    df = ids.copy()
    df["score"] = scores
    df["label"] = labels

    # Get true mapping
    true_mapping = get_mapping_from_labels(df, "label", is_label=True)

    # Get predicted ranking
    predicted_mapping = get_mapping_from_labels(df, "score", is_label=False)

    # Get top k predictions
    return mean_average_precision_at_k(true_mapping, predicted_mapping)


def evaluate_mapk_inference(model, X, ids, true_mapping):
    """Evaluate MAP@K. Assumes only one group per customer (one week_num)"""
    if ids.week_num.nunique() > 1:
        raise ValueError("Assumes only one group per customer (one week_num)")

    scores = model.predict(X)
    df = ids.copy()
    df["score"] = scores
    predicted_mapping = get_mapping_from_labels(df, "score", is_label=False)

    # Get top k predictions
    return mean_average_precision_at_k(true_mapping, predicted_mapping)


class RankerWithHyperparameterTuning:
    def __init__(self, config):
        """Initialize Ranker with configuration parameters.

        Args:
            config: Dictionary containing model and training parameters
        """
        self.feature_config = config["feature_config"]
        self.lightgbm_fixed_params = config["lightgbm_fixed_params"]
        self.study = None
        self.hyperparameters_config = config["hyperparameters_config"]
        self.n_trials = config["n_trials"]
        self.early_stopping_rounds = config["early_stopping_rounds"]
        self.metric_return = config["metric_return"]

    def get_all_features(self) -> List[str]:
        """Get list of all features to use."""
        all_features = []
        for feature_list in self.feature_config.values():
            all_features.extend(feature_list)
        return all_features

    def prepare_features(self, data: LightGBMDataResult) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for training/ validation."""
        logger.info("Preparing features for training/ validation")

        features = self.get_all_features()
        X, cols_features_categorical = prepare_features_for_ranker(data, features)

        logger.info("Features by domain:")
        for domain, features in self.feature_config.items():
            logger.info(f"{domain}: {len(features)} features")
            logger.info(f"{features}")

        return X, cols_features_categorical

    @staticmethod
    def _optimize_hyperparams(
        X_train,
        y_train,
        train_groups,
        cols_features_categorical,
        X_valid_train,
        y_valid_train,
        valid_groups,
        valid_train_ids,
        X_valid_inference,
        valid_inference_ids,
        valid_mapping,
        lightgbm_fixed_params,
        hyperparameters_config,
        train_sample_weight,
        val_sample_weight,
        n_trials,
        early_stopping_rounds,
        metric_return,
    ):
        """Optimize hyperparameters."""
        direction = "maximize"

        def objective(trial):
            with mlflow.start_run(nested=True):

                params = {
                    **lightgbm_fixed_params,
                }

                # Hyperparameters for tuning
                for param, config in hyperparameters_config.items():
                    if config["type"] == "float":
                        params[param] = trial.suggest_float(param, config["min"], config["max"])
                    elif config["type"] == "int":
                        params[param] = trial.suggest_int(param, config["min"], config["max"])
                    elif config["type"] == "categorical":
                        params[param] = trial.suggest_categorical(param, config["values"])

                callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=0), lgb.log_evaluation(period=0)]

                ranker = lgb.LGBMRanker(**params)
                ranker.fit(
                    X=X_train,
                    y=y_train,
                    group=train_groups,
                    categorical_feature=cols_features_categorical,
                    eval_set=[(X_train, y_train), (X_valid_train, y_valid_train)],
                    eval_group=[train_groups, valid_groups],
                    sample_weight=train_sample_weight,
                    eval_sample_weight=[train_sample_weight, val_sample_weight],
                    callbacks=callbacks,
                )

                # Replace with any metrics you need
                mapk_valid_train = evaluate_mapk_train(ranker, X_valid_train, valid_train_ids, y_valid_train)
                mapk_valid_inference = evaluate_mapk_inference(
                    ranker, X_valid_inference, valid_inference_ids, valid_mapping
                )

                # You can print anything in the middle for monitoring the process
                print("\nMAP@K (valid train): %.3f" % (mapk_valid_train))
                print("\nMAP@K (valid inference): %.3f" % (mapk_valid_inference))

                # Log to MLflow
                mlflow.log_params(params)
                mlflow.log_metric("mapk_valid_train", mapk_valid_train)
                mlflow.log_metric("mapk_valid_inference", mapk_valid_inference)

            return mapk_valid_inference if metric_return == "mapk_valid_inference" else mapk_valid_train

        study = optuna.create_study(direction=direction)
        # Specify the maximum number of seconds to run the study in timeout
        study.optimize(objective, show_progress_bar=True, n_trials=n_trials)

        return study

    def train(
        self,
        train_data: LightGBMDataResult,
        valid_train_data: LightGBMDataResult,
        valid_inference_data: LightGBMDataResult,
        valid_mapping: Dict[str, List[int]],
    ) -> None:
        """Train the ranking model.

        Args:
            train_data: Training data
        """
        logger.info("Training ranker")
        if train_data.use_type != "train":
            raise ValueError("Train data is not in train mode")
        if valid_train_data is not None and valid_train_data.use_type != "train":
            raise ValueError("Valid data is not in train mode")
        if valid_inference_data is not None and valid_inference_data.use_type != "inference":
            raise ValueError("Valid inference data is not in inference mode")

        # Prepare features and groups
        # Training data
        X_train, cols_features_categorical = self.prepare_features(train_data)
        y_train = train_data.label
        train_groups = train_data.group

        # Valid train data
        X_valid_train, _ = self.prepare_features(valid_train_data)
        y_valid_train = valid_train_data.label
        valid_groups = valid_train_data.group
        valid_train_ids = valid_train_data.data[["customer_id", "article_id", "week_num"]].copy()

        # Valid inference data
        X_valid_inference, _ = self.prepare_features(valid_inference_data)
        valid_inference_ids = valid_inference_data.data[["customer_id", "article_id", "week_num"]].copy()

        with logging_redirect_tqdm():
            study = self._optimize_hyperparams(
                X_train=X_train,
                y_train=y_train,
                train_groups=train_groups,
                cols_features_categorical=cols_features_categorical,
                X_valid_train=X_valid_train,
                y_valid_train=y_valid_train,
                valid_groups=valid_groups,
                valid_train_ids=valid_train_ids,
                X_valid_inference=X_valid_inference,
                valid_inference_ids=valid_inference_ids,
                valid_mapping=valid_mapping,
                lightgbm_fixed_params=self.lightgbm_fixed_params,
                hyperparameters_config=self.hyperparameters_config,
                train_sample_weight=None,
                val_sample_weight=None,
                n_trials=self.n_trials,
                early_stopping_rounds=self.early_stopping_rounds,
                metric_return=self.metric_return,
            )
        return study


@dataclass
class RankerHyperparameterTuningConfig:
    """Configuration for hyperparameter tuning pipeline."""

    # I/O
    sample: str = "train"  # train, valid
    subsample: float = 0.05
    seed: int = 42

    # Experiment tracking
    tag: str = "ranker-hyperparameter-tuning"
    experiment_name: str = "fashion_recommendation"

    # Hyperparameter tuning config
    hyperparameter_config: Dict[str, Any] = field(
        default_factory=lambda: get_default_config_ranker_with_hyperparameter_tuning()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RankerHyperparameterTuningConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def get_default_config(cls) -> "RankerHyperparameterTuningConfig":
        return cls.from_dict(DEFAULT_RANKER_HYPERPARAMETERS_PIPELINE_CONFIG)


class RankerHyperparameterTuningPipeline:
    """Pipeline for hyperparameter tuning of ranker models."""

    def __init__(self, config: Optional[RankerHyperparameterTuningConfig] = None):
        """Initialize pipeline with config."""
        self.config = config or RankerHyperparameterTuningConfig()
        self.tuner = None
        self.best_params = None
        self.study = None
        self.best_trial = None
        self.experiment_name = self.config.experiment_name
        self.run_id = None
        self.id = str(int(datetime.now().timestamp()))
        prefix = "hyperparameter_tuning_" + self.config.sample
        if self.config.subsample < 1:
            prefix += f"_subsample_{self.config.subsample}_{self.config.seed}"
        else:
            prefix += "_full"
        self.run_name = f"{prefix}_{self.id}"

    def setup(self, config: Optional[RankerHyperparameterTuningConfig] = None) -> "RankerHyperparameterTuningPipeline":
        """Set up the pipeline with configuration."""
        logger.info("Setting up RankerHyperparameterTuningPipeline")
        if config is not None:
            self.config = config

        logger.info(f"Using configuration: {json.dumps(self.config.to_dict(), indent=2)}")
        self.tuner = RankerWithHyperparameterTuning(self.config.hyperparameter_config)
        return self

    def _load_data(self) -> Tuple[LightGBMDataResult, LightGBMDataResult, LightGBMDataResult, Dict]:
        """Load data for training and validation."""
        logger.info("Loading data for hyperparameter tuning")

        # Load training data
        path_to_lightgbm_data_train = get_path_to_lightgbm_data(
            sample="train",
            use_type="train",
            subsample=self.config.subsample,
            seed=self.config.seed,
        )
        train_data = LightGBMDataResult.load(path_to_lightgbm_data_train)
        logger.info(f"Loaded training data from {path_to_lightgbm_data_train}")

        # Load validation data (train mode)
        path_to_lightgbm_data_valid_train = get_path_to_lightgbm_data(
            sample="valid",
            use_type="train",
            subsample=self.config.subsample,
            seed=self.config.seed,
        )
        valid_train_data = LightGBMDataResult.load(path_to_lightgbm_data_valid_train)
        logger.info(f"Loaded validation (train mode) data from {path_to_lightgbm_data_valid_train}")

        # Load validation data (inference mode)
        path_to_lightgbm_data_valid_inference = get_path_to_lightgbm_data(
            sample="valid",
            use_type="inference",
            subsample=self.config.subsample,
            seed=self.config.seed,
        )
        valid_inference_data = LightGBMDataResult.load(path_to_lightgbm_data_valid_inference)
        logger.info(f"Loaded validation (inference mode) data from {path_to_lightgbm_data_valid_inference}")

        # Load validation mapping
        valid_mapping = load_optimized_raw_data(
            data_type="candidates_to_articles_mapping",
            sample="valid",
            subsample=self.config.subsample,
            seed=self.config.seed,
        )
        logger.info("Loaded validation mapping")

        return train_data, valid_train_data, valid_inference_data, valid_mapping

    def run(self) -> Tuple[Dict, pd.DataFrame, str]:
        """Run the hyperparameter tuning pipeline.

        Returns:
            Tuple containing:
            - Dictionary of evaluation metrics
            - DataFrame of feature importance
            - MLflow run ID
        """
        if self.tuner is None:
            raise ValueError("Pipeline is not set up. Call setup() before running the pipeline.")

        # Load data
        train_data, valid_train_data, valid_inference_data, valid_mapping = self._load_data()

        # Set up MLflow
        setup_mlflow(self.experiment_name)

        # Start MLflow run
        with mlflow.start_run(run_name=self.run_name, nested=True) as run:

            # Run hyperparameter tuning
            logger.info("Starting hyperparameter tuning")
            self.study = self.tuner.train(
                train_data=train_data,
                valid_train_data=valid_train_data,
                valid_inference_data=valid_inference_data,
                valid_mapping=valid_mapping,
            )

            # Get best trial and parameters
            self.best_trial = self.study.best_trial
            self.best_params = self.best_trial.params
            logger.info(f"Best trial: {self.best_trial.number}")
            logger.info(f"Best value: {self.best_trial.value}")
            logger.info(f"Best parameters: {self.best_params}")

            # Log best parameters
            log_model_params(self.best_params)
            log_model_params(self.config.hyperparameter_config["lightgbm_fixed_params"])

            self.run_id = run.info.run_id
            logger.info(f"MLflow run ID: {self.run_id}")

            # Log config
            log_config(self.config)

            return self.run_id
