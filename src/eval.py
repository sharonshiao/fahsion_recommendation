import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_RANKER_EVALUATOR_CONFIG,
    DEFAULT_TEST_WEEK_NUM,
    DEFAULT_VALID_WEEK_NUM,
)
from src.experiment_tracking import setup_mlflow
from src.feature_extraction import load_optimized_raw_data
from src.input_preprocessing import (
    LightGBMDataResult,
    get_path_to_lightgbm_data,
)
from src.metrics import (
    ideal_mean_average_precision_at_k,
    mean_average_precision_at_k,
)
from src.ranker import Ranker
from src.utils.popularity import calculate_rolling_popular_items

logger = logging.getLogger(__name__)


@dataclass
class RankerEvaluatorConfig:
    """Configuration for the Ranker Evaluator."""

    # I/O
    sample: str = "valid"  # valid, test
    subsample: float = 0.05
    seed: int = 42

    # Evaluation parameters
    config_evaluator: dict = field(
        default_factory=lambda: {"k": 12, "heuristic_strategy": "rolling_popular_items", "test_sample": ["valid"]}
    )

    # Experiment tracking
    experiment_name: str = "fashion_recommendation"
    run_id: str = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RankerEvaluatorConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def get_default_config(cls) -> "RankerEvaluatorConfig":
        return RankerEvaluatorConfig(**DEFAULT_RANKER_EVALUATOR_CONFIG)


class RankerEvaluator:
    """Class for evaluating ranking models."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the evaluator with configuration.

        Args:
            config: Dictionary containing evaluation parameters
        """
        self.config = config
        self.k = config.get("k", 12)
        self.heuristic_strategy = config.get("heuristic_strategy", "rolling_popular_items")
        self.test_sample = config.get("test_sample", "valid")
        if self.test_sample == "valid":
            self.test_week_num = DEFAULT_VALID_WEEK_NUM
        elif self.test_sample == "test":
            self.test_week_num = DEFAULT_TEST_WEEK_NUM
        else:
            raise ValueError(f"Invalid test sample: {self.test_sample}")

    @staticmethod
    def _calculate_heuristic_prediction(
        transactions: pd.DataFrame, week_num: int, k: int, heuristic_strategy: str
    ) -> dict:
        """Calculate the heuristic prediction."""
        logger.info(f"Calculating heuristic prediction for week {week_num} with strategy {heuristic_strategy}")
        if heuristic_strategy == "rolling_popular_items":
            num_rolling_weeks = 1
            heuristic_pred = (
                calculate_rolling_popular_items(
                    transactions.query("week_num <= @week_num - 1 and week_num >= @week_num - @num_rolling_weeks"),
                    num_rolling_weeks,
                    k,
                    "week_num",
                    "article_id",
                )
                .query("week_num == @week_num - 1")["article_id"]
                .to_list()
            )
        else:
            raise NotImplementedError(f"Heuristic strategy {heuristic_strategy} not implemented")

        logger.debug(f"Heuristic prediction: {heuristic_pred}")
        return heuristic_pred

    def plot_feature_importance(self, ranker: Ranker):
        """Plot the feature importance."""
        logger.info(f"Plotting feature importance")
        feature_importance = ranker.get_feature_importance()
        print(feature_importance)
        # plot feature importance
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.barh(feature_importance.feature, feature_importance.importance)
        ax.set_title("Feature Importance")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Importance")
        return fig, ax

    def evaluate(
        self, ranker: Ranker, test_inference_data: LightGBMDataResult, test_mapping: dict, transactions: pd.DataFrame
    ) -> float:
        """Evaluate the ranker model."""
        logger.info(f"Evaluating ranker model on test data")

        # Calculate MAP@K for model prediction
        logger.info(f"Calculating MAP@K for model prediction")
        preds_model = ranker.predict_ranks(test_inference_data)
        mapk_model = mean_average_precision_at_k(test_mapping, preds_model, self.k)

        # Calculate MAP@K for heuristic prediction
        logger.info(f"Calculating MAP@K for heuristic prediction")
        heuristic_articles = self._calculate_heuristic_prediction(
            transactions, self.test_week_num, self.k, self.heuristic_strategy
        )
        preds_heuristic = {k: heuristic_articles for k in test_mapping.keys()}
        mapk_heuristic = mean_average_precision_at_k(test_mapping, preds_heuristic, self.k)

        # Calculate MAP@K for ideal prediction
        logger.info(f"Calculating MAP@K for ideal prediction")
        mapk_ideal = ideal_mean_average_precision_at_k(test_mapping, preds_model, self.k)

        # # Plot feature importance
        # logger.info(f"Plotting feature importance")
        # fig, ax = self._plot_feature_importance(ranker)
        # plt.show()

        return {
            "mapk_model": mapk_model,
            "mapk_heuristic": mapk_heuristic,
            "mapk_ideal": mapk_ideal,
        }


class RankerEvaluatorPipeline:
    """Pipeline for evaluating ranker models."""

    def __init__(self, config: Optional[RankerEvaluatorConfig] = None):
        """Initialize pipeline with config."""
        self.config = config or RankerEvaluatorConfig()
        self.ranker = None
        self.evaluator = {}
        self.results = None
        self.experiment_name = self.config.experiment_name
        self.run_id = self.config.run_id
        self.sample = self.config.sample
        self.subsample = self.config.subsample
        self.seed = self.config.seed
        self.data = None
        self.mapping = None
        self.transactions = None

    def _load_ranker(self, run_id: str):
        """Load the ranker model."""
        logger.info(f"Loading ranker model")
        setup_mlflow(self.experiment_name)
        model_uri = f"runs:/{run_id}/model"
        self.ranker = mlflow.lightgbm.load_model(model_uri)

    def _load_test_data(self):
        """Load the test data."""
        logger.info(f"Loading data: {self.config.sample}")

        data = {}
        mapping = {}

        for sample in self.sample:
            test_inference_data_path = get_path_to_lightgbm_data(
                sample, "inference", self.config.subsample, self.config.seed
            )
            test_inference_data = LightGBMDataResult.load(test_inference_data_path)
            data[sample] = test_inference_data
            mapping[sample] = load_optimized_raw_data(
                data_type="candidates_to_articles_mapping",
                sample=sample,
                subsample=self.config.subsample,
                seed=self.config.seed,
            )

        self.data = data
        self.mapping = mapping

    def _load_transactions(self):
        """Load the transactions data."""
        logger.info(f"Loading transactions data")
        transactions_train = load_optimized_raw_data(
            data_type="transactions", sample="train", subsample=self.config.subsample, seed=self.config.seed
        )
        transactions_valid = load_optimized_raw_data(
            data_type="transactions", sample="valid", subsample=self.config.subsample, seed=self.config.seed
        )
        transactions_test = load_optimized_raw_data(
            data_type="transactions", sample="test", subsample=self.config.subsample, seed=self.config.seed
        )
        transactions = pd.concat([transactions_train, transactions_valid, transactions_test], axis=0, ignore_index=True)
        # Clean up memory
        del transactions_train, transactions_valid, transactions_test
        self.transactions = transactions

    def setup(self):
        """Setup the pipeline."""
        logger.info(f"Setting up ranker evaluator pipeline")
        logger.debug(f"Config: {json.dumps(self.config.to_dict(), indent=2)}")

        # Set up evaluator
        for sample in self.sample:
            tmp_config = self.config.config_evaluator.copy()
            tmp_config["test_sample"] = sample
            self.evaluator[sample] = RankerEvaluator(tmp_config)
        logger.info(f"Evaluator setup complete")

    def run(self) -> float:
        """Run the pipeline to evaluate the ranker model."""
        logger.info(f"Running ranker evaluator pipeline")

        # Load ranker model
        self._load_ranker(self.run_id)

        # Load data
        self._load_test_data()
        self._load_transactions()

        # Evaluate
        results = {}
        for sample in self.sample:
            results[sample] = self.evaluator[sample].evaluate(
                self.ranker, self.data[sample], self.mapping[sample], self.transactions
            )

        # Plot feature importance
        logger.info(f"Plotting feature importance")
        fig, ax = self.evaluator[self.sample[0]].plot_feature_importance(self.ranker)
        ax.set_title(f"Feature Importance for {self.run_id}")
        plt.show()

        self.results = results

        return results
