import json
import logging
import os
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from mlflow.models import infer_signature

from src.config import DEFAULT_RANKER_PIPELINE_CONFIG
from src.experiment_tracking import (
    log_config,
    log_feature_importance,
    log_model_params,
    setup_mlflow,
)
from src.feature_extraction import load_optimized_raw_data
from src.input_preprocessing import (
    LightGBMDataResult,
    get_path_to_lightgbm_data,
)
from src.metrics import (
    get_mapping_from_labels,
    mean_average_precision_at_k,
    mean_average_precision_at_k_hierarchical,
)

logger = logging.getLogger(__name__)


@dataclass
class RankerConfig:
    """Configuration for the Ranker model."""

    # I/O
    sample: str = "train"  # train, valid
    subsample: float = 0.05
    seed: int = 42

    # Experiment tracking
    experiment_name: str = "fashion_recommendation"
    tag: str = "ranker-baseline"

    # Feature configuration
    feature_config: Dict[str, Any] = field(
        default_factory=lambda: {
            # Feature lists by domain
            "article_features": [
                "product_type_no",
                "graphical_appearance_no",
                "colour_group_code",
                "perceived_colour_value_id",
                "perceived_colour_master_id",
                "department_no",
                "index_code",
                "index_group_no",
                "section_no",
                "garment_group_no",
            ],
            "customer_features": [
                "age_bin",
                "club_member_status",
                "fashion_news_frequency",
                "fn",
                "active",
                "postal_code",
            ],
            "transaction_features": [
                "bestseller_rank",
            ],
        }
    )

    # LGBM ranker parameters
    lightgbm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "ranker_params": {
                "objective": "lambdarank",
                "metrics": "ndcg",
                "boosting_type": "dart",
                "n_estimators": 1,
                "importance_type": "gain",
                "verbose": 10,
                "random_state": 123,
            },
            "fit_params": {"early_stopping_rounds": 3},
            "use_validation_set": False,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RankerConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def save(self, path_to_dir: str):
        """Save config to directory."""
        logger.info(f"Saving config to {path_to_dir}")
        os.makedirs(path_to_dir, exist_ok=True)
        with open(path_to_dir + f"/config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Config saved to {path_to_dir}")

    @classmethod
    def load(cls, path_to_dir: str) -> "RankerConfig":
        """Load config from directory."""
        logger.info(f"Loading config from {path_to_dir}")
        with open(path_to_dir + f"/config.json", "r") as f:
            config = cls.from_dict(json.load(f))
            logger.info(f"Config loaded from {path_to_dir}")
            return config

    @classmethod
    def get_default_config(cls) -> "RankerConfig":
        return cls.from_dict(DEFAULT_RANKER_PIPELINE_CONFIG)


class Ranker:
    def __init__(self, feature_config: Dict, lightgbm_params: Dict):
        """Initialize Ranker with configuration parameters.

        Args:
            config: Dictionary containing model and training parameters
        """
        self.feature_config = feature_config
        self.lightgbm_params = lightgbm_params
        self.model = None

    def get_all_features(self) -> List[str]:
        """Get list of all features to use."""
        all_features = []
        for feature_list in self.feature_config.values():
            all_features.extend(feature_list)
        return all_features

    def prepare_features(self, data: LightGBMDataResult) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for training/ validation."""
        logger.info(f"Preparing features for training/ validation")

        # Get all features we want to use
        cols_features = self.get_all_features()

        # Only include features that are also in df
        cols_features_available = data.get_feature_names_list()
        for v in cols_features:
            if v not in cols_features_available:
                raise ValueError(f"Feature {v} not found in data")

        # Get categorical features from data's feature names
        cols_features_categorical = [col for col in cols_features if col in data.feature_names["categorical_features"]]

        logger.info(f"Number of features: {len(cols_features)}")
        logger.info(f"Features by domain:")
        for domain, features in self.feature_config.items():
            logger.info(f"{domain}: {len(features)} features")
            logger.info(f"{features}")
        logger.info(f"Number of features categorical: {len(cols_features_categorical)}")
        logger.info(f"Cols features categorical: {cols_features_categorical}")

        return data.data[cols_features], cols_features_categorical

    def train(self, train_data: LightGBMDataResult, valid_data: Optional[LightGBMDataResult] = None) -> None:
        """Train the ranking model.

        Args:
            train_data: Training data
        """
        logger.info(f"Training ranker")
        if train_data.use_type != "train":
            raise ValueError("Train data is not in train mode")
        if valid_data is not None and valid_data.use_type != "train":
            raise ValueError("Valid data is not in train mode")

        # Prepare features and groups
        X_train, cols_features_categorical = self.prepare_features(train_data)
        y_train = train_data.label
        train_groups = train_data.group

        # Initialize model
        logger.info(f"Initializing model")
        self.model = LGBMRanker(
            **self.lightgbm_params["ranker_params"],
        )

        logger.info(f"Training model")
        metrics = {}
        if not self.lightgbm_params["use_validation_set"]:
            # Train model
            self.model.fit(
                X=X_train,
                y=y_train,
                group=train_groups,
                categorical_feature=cols_features_categorical,
            )

        else:
            X_valid, _ = self.prepare_features(valid_data)
            y_valid = valid_data.label
            valid_groups = valid_data.group

            # Train model
            self.model.fit(
                X=X_train,
                y=y_train,
                group=train_groups,
                categorical_feature=cols_features_categorical,
                eval_set=[(X_valid, y_valid)],
                eval_group=[valid_groups],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=self.lightgbm_params["fit_params"]["early_stopping_rounds"])
                ],
            )

        # Infer signature
        self.signature = infer_signature(X_train, self.model.predict(X_train))

        logger.info(f"Model trained")

    def predict_scores(self, data: LightGBMDataResult) -> np.ndarray:
        """Predict scores"""
        logger.info(f"Predicting scores")
        if data.use_type != "inference":
            warnings.warn("Data is not in inference mode")

        if self.model is None:
            raise ValueError("Model not trained yet")

        X_test, _ = self.prepare_features(data)
        logger.info(f"Completed prediction")
        return self.model.predict(X_test)

    def predict_ranks(self, data: LightGBMDataResult) -> dict[str, List[int]]:
        """Predict ranks for each group. Assume there is only one gorup per customer."""
        logger.info(f"Predicting ranks")
        if data.use_type != "inference":
            warnings.warn("Data is not in inference mode")

        if self.model is None:
            raise ValueError("Model not trained yet")

        # Get predictions
        scores = self.predict_scores(data)

        # Add scores to data
        results = data.data[["customer_id", "article_id"]].copy()
        results["score"] = scores

        # Sort by customer_id and score (descending) to get ranks
        results = results.sort_values(["customer_id", "score"], ascending=[True, False])
        logger.info(f"Completed prediction")
        return results.groupby("customer_id")["article_id"].apply(list).to_dict()

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model.

        Returns:
            DataFrame with feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        importance = pd.DataFrame({"feature": self.get_all_features(), "importance": self.model.feature_importances_})
        importance["importance"] = importance["importance"] / importance["importance"].sum()
        return importance.sort_values("importance", ascending=False)

    def save(self, path_to_dir: str):
        """Save the ranker model and its configuration to disk.

        Args:
            path_to_dir: Directory path to save the model and config
        """
        logger.info(f"Saving ranker to {path_to_dir}")

        if self.model is None:
            raise ValueError("Model not trained yet. Call train() before saving.")

        # Create directory if it doesn't exist
        os.makedirs(path_to_dir, exist_ok=True)

        # Save the model using LightGBM's native method
        path_model = os.path.join(path_to_dir, "model.txt")
        self.model.booster_.save_model(path_model)
        logger.info(f"Model saved to {path_model}")

        # Save configuration
        config = {"feature_config": self.feature_config, "lightgbm_params": self.lightgbm_params}
        with open(os.path.join(path_to_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {path_to_dir}/config.json")

    @classmethod
    def load(cls, path_to_dir: str) -> "Ranker":
        """Load a ranker model from disk.

        Args:
            path_to_dir: Directory path containing the saved model and config

        Returns:
            Loaded Ranker instance
        """
        logger.info(f"Loading ranker from {path_to_dir}")

        # Load configuration
        with open(os.path.join(path_to_dir, "config.json"), "r") as f:
            config = json.load(f)

        # Create ranker instance
        ranker = cls(feature_config=config["feature_config"], lightgbm_params=config["lightgbm_params"])

        # Load the model
        path_model = os.path.join(path_to_dir, "model.txt")
        if os.path.exists(path_model):
            ranker.model = lgb.Booster(model_file=path_model)
            logger.info(f"Model loaded from {path_model}")
        else:
            logger.warning(f"No model file found at {path_model}")

        return ranker


class RankerTrainValidPipeline:
    """End-to-end pipeline for training and validating the ranker model."""

    def __init__(self, config: Optional[RankerConfig] = None):
        """Initialize pipeline with config."""
        self.config = config or RankerConfig()
        self.ranker = None
        self.id = str(int(datetime.now().timestamp()))
        prefix = self.config.sample
        if self.config.subsample < 1:
            prefix += f"_subsample_{self.config.subsample}_{self.config.seed}"
        else:
            prefix += "_full"
        self.run_name = f"{prefix}_{self.id}"

    def setup(self, config: Optional[RankerConfig] = None) -> "RankerTrainValidPipeline":
        """Set up the pipeline with configuration."""
        logger.info("Setting up RankerTrainValidPipeline")
        if config is not None:
            self.config = config
        if self.config is None:
            raise ValueError("Config must be provided")
        logger.debug(f"Pipeline config: {json.dumps(self.config.to_dict(), indent=2)}")
        self.ranker = Ranker(feature_config=self.config.feature_config, lightgbm_params=self.config.lightgbm_params)
        return self

    def _load_data(self) -> Tuple[LightGBMDataResult, Optional[LightGBMDataResult]]:
        """Load data for training and validation."""
        logger.info("Loading data for RankerTrainValidPipeline")

        if self.config.sample == "train":
            path_train = get_path_to_lightgbm_data("train", "train", self.config.subsample, self.config.seed)
            path_valid_train = get_path_to_lightgbm_data("valid", "train", self.config.subsample, self.config.seed)
            path_valid_inference = get_path_to_lightgbm_data(
                "test", "inference", self.config.subsample, self.config.seed
            )
        elif self.config.sample == "valid":
            path_train = get_path_to_lightgbm_data("valid", "train", self.config.subsample, self.config.seed)
            path_valid_train = get_path_to_lightgbm_data("test", "train", self.config.subsample, self.config.seed)
            path_valid_inference = get_path_to_lightgbm_data(
                "test", "inference", self.config.subsample, self.config.seed
            )
        else:
            raise ValueError(f"Invalid sample type: {self.config.sample}")

        train_data = LightGBMDataResult.load(path_train)
        valid_inference_data = LightGBMDataResult.load(path_valid_inference)
        valid_train_data = None
        if self.config.lightgbm_params["use_validation_set"]:
            valid_train_data = LightGBMDataResult.load(path_valid_train)

        return train_data, valid_train_data, valid_inference_data

    def _save_ranker(self):
        if self.ranker is None or self.ranker.model is None:
            raise ValueError("Model not trained yet. Call train() before saving.")

        path_model = "model.txt"
        self.ranker.model.booster_.save_model(path_model)
        mlflow.log_artifact(path_model)
        logger.info(f"Model saved to {path_model}")
        logger.info(f"Ranker saved via mlflow")

    def _evaluate_model_on_train_data(self, data: LightGBMDataResult):
        scores = self.ranker.predict_scores(data)
        data_df = data.data[["customer_id", "week_num", "article_id", "label"]].copy()
        data_df["score"] = scores
        data_df.sort_values(["customer_id", "week_num", "score"], ascending=[True, True, False], inplace=True)
        data_df.reset_index(drop=True, inplace=True)

        # Get predicted ranking
        preds = get_mapping_from_labels(data_df, "score", is_label=False)

        # Get true mapping
        true_mapping = get_mapping_from_labels(data_df, "label", is_label=True)

        # Calculate MAP@K
        if data_df.week_num.nunique() > 1:
            mapk = mean_average_precision_at_k_hierarchical(true_mapping, preds, k=12)
        else:
            mapk = mean_average_precision_at_k(true_mapping, preds, k=12)
        return mapk

    def _evaluate_model_on_inference_data(self, valid_inference_data: LightGBMDataResult) -> dict:
        # This assumes we only have one week in the inference data
        logger.info(f"Evaluating model on inference data")
        # Load inference data
        if self.config.sample == "train":
            sample_inference = "valid"
        elif self.config.sample == "valid":
            sample_inference = "test"

        path_valid_inference = get_path_to_lightgbm_data(
            sample_inference, "inference", self.config.subsample, self.config.seed
        )
        valid_inference_data = LightGBMDataResult.load(path_valid_inference)
        if valid_inference_data.data.week_num.nunique() > 1:
            raise ValueError("Inference data has multiple weeks")

        # Get predictions
        preds = self.ranker.predict_ranks(valid_inference_data)

        # Get true mapping
        true_mapping = load_optimized_raw_data(
            data_type="candidates_to_articles_mapping",
            sample=sample_inference,
            subsample=self.config.subsample,
            seed=self.config.seed,
        )

        # Calculate MAP@K
        mapk = mean_average_precision_at_k(true_mapping, preds, k=12)
        return mapk

    def evaluate(
        self,
        train_data: LightGBMDataResult,
        valid_inference_data: LightGBMDataResult,
        valid_train_data: Optional[LightGBMDataResult] = None,
    ):
        mapk_train = self._evaluate_model_on_train_data(train_data)
        mapk_valid_inference = self._evaluate_model_on_inference_data(valid_inference_data)
        if self.config.lightgbm_params["use_validation_set"]:
            mapk_valid_train = self._evaluate_model_on_train_data(valid_train_data)
        else:
            mapk_valid_train = None

        metrics = {
            "mapk_train": mapk_train,
            "mapk_valid_inference": mapk_valid_inference,
        }
        if self.config.lightgbm_params["use_validation_set"]:
            metrics["mapk_valid"] = mapk_valid_train
        return metrics

    def run(self) -> pd.DataFrame:
        """Run the pipeline to train and validate the model.

        Returns:
            Tuple containing:
            - Feature importance DataFrame
        """
        if self.ranker is None:
            raise ValueError("Pipeline not set up. Call setup() before running the pipeline.")

        # Load data
        train_data, valid_train_data, valid_inference_data = self._load_data()

        # Set up MLflow experiment
        setup_mlflow(self.config.experiment_name)
        mlflow.lightgbm.autolog()

        with mlflow.start_run(run_name=self.run_name) as current_run:

            # Training
            self.ranker.train(train_data, valid_train_data)

            # Get feature importance
            feature_importance = self.ranker.get_feature_importance()

            # Evaluate performance on valid inference set
            metrics = self.evaluate(train_data, valid_train_data, valid_inference_data)
            mlflow.log_metrics(metrics)

            # Log configuration
            log_config(self.config)

            # Log params
            log_feature_importance(feature_importance["feature"].tolist(), feature_importance["importance"].values)

            # Log model parameters
            log_model_params(self.config.lightgbm_params["ranker_params"])

            # Set tag
            mlflow.set_tag("tag", self.config.tag)

            # Log model
            mlflow.lightgbm.log_model(self.ranker, "model", signature=self.ranker.signature)

            # Save log file
            mlflow.log_artifact("train_ranker.log")

            # Save model
            self._save_ranker()

            return metrics, feature_importance, current_run.info.run_id
