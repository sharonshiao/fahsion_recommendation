"""MLflow experiment tracking utilities."""

import logging
from dataclasses import asdict
from typing import Any, Dict, List

import mlflow
import numpy as np

logger = logging.getLogger(__name__)

MODEL_REGISTRY = "../model/mlruns"


def setup_mlflow(experiment_name: str) -> None:
    """Set up MLflow experiment.

    Args:
        experiment_name: Name of the experiment
    """
    # Set tracking URI to local mlruns directory
    mlflow.set_tracking_uri(MODEL_REGISTRY)

    # Create experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)


def log_model_params(params: Dict[str, Any]) -> None:
    """Log model parameters to MLflow.

    Args:
        params: Dictionary of parameters to log
    """
    mlflow.log_params(params)


def log_feature_importance(feature_names: List[str], importance_values: np.ndarray) -> None:
    """Log feature importance plot and values.

    Args:
        feature_names: List of feature names
        importance_values: Array of feature importance values
    """
    import matplotlib.pyplot as plt

    # Create feature importance plot
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(importance_values)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    plt.barh(pos, importance_values[sorted_idx])
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance (Top 20)")

    # Save plot as artifact
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()

    # Log feature importance as a parameter
    importance_dict = dict(zip(feature_names, importance_values.tolist()))
    mlflow.log_dict(importance_dict, "feature_importance.json")


def log_config(config: Any) -> None:
    """Log configuration to MLflow.

    Args:
        config: Configuration object with to_dict method
    """
    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()
    else:
        config_dict = asdict(config)

    mlflow.log_dict(config_dict, "config.json")
