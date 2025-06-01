import json
import logging
import os
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

from src.input_preprocessing import LightGBMDataResult
from src.ranker import Ranker

logger = logging.getLogger(__name__)


# FIX: This might be redundant
# Maybe useful for test inference set
class RankerEvaluator:
    def __init__(self, config: dict):
        self.config = config

    @staticmethod
    def apk(actual, predicted, k=12):
        """Computes the average precision at k.

        This function computes the average precision at k between two lists of
        items.

        Parameters
        ----------
        actual : list
                A list of elements that are to be predicted (order doesn't matter)
        predicted : list
                    A list of predicted elements (order does matter)
        k : int, optional
            The maximum number of predicted elements

        Returns
        -------
        score : double
                The average precision at k over the input lists

        """
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            return 0.0

        return score / min(len(actual), k)

    @staticmethod
    def mean_average_precision_at_k(
        actual: dict, predicted: dict, k: int = 12, default_prediction: np.ndarray = None
    ) -> float:
        """Calculate mean average precision at k."""
        logger.info(f"Evaluating ranking")
        apks = []
        if default_prediction is None:
            default_prediction = []
        for c_id, gt in actual.items():
            pred = predicted.get(c_id, [])
            pred = np.concatenate([np.array(pred), default_prediction])
            apks.append(RankerEvaluator.apk(gt, pred[:k]))
        logger.info(f"Mean average precision at k: {np.mean(apks)}")
        return np.mean(apks)

    def evaluate(self, ranker: Ranker, test_inference_data: LightGBMDataResult, test_mapping: dict) -> float:
        """Evaluate the ranker model."""
        logger.info(f"Evaluating ranker model on test data")
        predictions = ranker.predict_ranks(test_inference_data)
        default_prediction = self.config.get("default_prediction", test_inference_data.default_prediction)
        return self.mean_average_precision_at_k(test_mapping, predictions, self.config.get("k", 12), default_prediction)


# FIX: This might be redundant
# Maybe useful for test inference set
class RankerEvaluatorPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.id = str(int(datetime.now().timestamp()))
        self.results = None

    def _get_path_to_dir(self) -> str:
        """Get path to directory for saving evaluation results."""
        path_to_dir = f"../evaluation/ranker/{self.ranker.id}/{self.id}"
        return path_to_dir

    def save(self):
        """Save evaluation results and configuration to disk."""
        if self.results is None:
            raise ValueError("No results to save. Run evaluation first.")

        # Create directory if it doesn't exist
        path_to_dir = self._get_path_to_dir()
        os.makedirs(path_to_dir, exist_ok=True)
        logger.info(f"Saving evaluation results to {path_to_dir}")

        # Save configuration
        with open(os.path.join(path_to_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to {path_to_dir}/config.json")

        # Save results
        results_dict = {
            "map_k": self.results,
            "timestamp": self.id,
            "ranker_path": self.config["ranker_path"],
            "ranker_id": self.ranker.id,
        }
        with open(os.path.join(path_to_dir, "results.json"), "w") as f:
            json.dump(results_dict, f, indent=2)
        logger.info(f"Results saved to {path_to_dir}/results.json")

    @classmethod
    def load_results(cls, path_to_dir: str) -> Tuple[dict, dict]:
        """Load evaluation results and configuration from disk.

        Args:
            path_to_dir: Directory containing the saved results

        Returns:
            Tuple of (config dict, results dict)
        """
        logger.info(f"Loading evaluation results from {path_to_dir}")

        # Load configuration
        with open(os.path.join(path_to_dir, "config.json"), "r") as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {path_to_dir}/config.json")

        # Load results
        with open(os.path.join(path_to_dir, "results.json"), "r") as f:
            results = json.load(f)
        logger.info(f"Results loaded from {path_to_dir}/results.json")

        return config, results

    def _load_test_data(self):
        """Load the test data."""
        logger.info(f"Loading test data")
        test_inference_data = LightGBMDataResult.load(self.config["test_data"]["data_path"])
        test_mapping = json.load(open(self.config["test_data"]["mapping_path"]))
        test_mapping = {int(k): v for k, v in test_mapping.items()}
        return test_inference_data, test_mapping

    def run(self) -> float:
        """Run the pipeline to evaluate the ranker model."""
        logger.info(f"Running ranker evaluator pipeline")
        logger.debug(f"Config: {json.dumps(self.config, indent=2)}")

        self.ranker = Ranker.load(self.config["ranker_path"])
        self.ranker.id = self.config["ranker_path"].split("/")[-1]

        # Load data
        test_inference_data, test_mapping = self._load_test_data()

        # Set up evaluator
        evaluator = RankerEvaluator(self.config["config_evaluator"])

        # Evaluate
        results = evaluator.evaluate(self.ranker, test_inference_data, test_mapping)
        self.results = results

        # Save results
        self.save()

        return results
