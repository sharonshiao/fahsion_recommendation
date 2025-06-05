import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_mapping_from_labels(raw_data: pd.DataFrame, col_score: str, is_label: bool = True) -> dict:
    """Get mapping from labels. considers the order of the labels."""
    if "customer_id" not in raw_data.columns:
        raise ValueError("customer_id column not found in data")
    if "article_id" not in raw_data.columns:
        raise ValueError("article_id column not found in data")
    if col_score not in raw_data.columns:
        raise ValueError(f"label column not found in data")

    # Sort by score
    if is_label:
        data = raw_data[raw_data[col_score] == 1].copy()  # Create an explicit copy
    else:
        data = raw_data.copy()  # Create an explicit copy

    # Sort by score (descending) by customer_id and week_num
    # Also sort by article_id to ensure the order is deterministic
    data.sort_values(
        ["customer_id", "week_num", col_score, "article_id"], ascending=[True, True, False, True], inplace=True
    )
    data.reset_index(drop=True, inplace=True)

    # If this contains multiple weeks, we will return a hierarchical dict
    num_weeks = len(data["week_num"].unique())
    if num_weeks > 1:
        map = {}
        for week in data["week_num"].unique():
            week_df = data[data["week_num"] == week].copy()  # Create an explicit copy
            map[week] = week_df.groupby("customer_id").article_id.apply(list).to_dict()
    else:
        map = data.groupby("customer_id").article_id.apply(list)
    return map


def average_precision_at_k(actual, predicted, k=12):
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


def mean_average_precision_at_k(map_true: dict, map_pred: dict, k: int = 12) -> float:
    """Calculate mean average precision at k."""
    logger.info(f"Evaluating ranking")
    apks = []
    for c_id, gt in map_true.items():
        pred = map_pred.get(c_id, [])
        apks.append(average_precision_at_k(gt, pred[:k]))
    logger.info(f"Mean average precision at k: {np.mean(apks)}")
    return np.mean(apks)


def mean_average_precision_at_k_hierarchical(map_true: dict, map_pred: dict, k: int = 12) -> float:
    """Calculate mean average precision at k for hierarchical mapping."""
    logger.info(f"Evaluating ranking")
    apks = []
    sum_obs = 0
    for week in map_true.keys():
        n = len(map_true[week])
        logger.debug(f"Week {week}, number of obseravtions: {n}")
        map = mean_average_precision_at_k(map_true[week], map_pred[week], k)
        apks.append(map * n)
        sum_obs += n
    logger.info(f"Mean average precision atk: {np.sum(apks) / sum_obs}")
    return np.sum(apks) / sum_obs


def ideal_average_precision_at_k(actual, predicted, k=12):
    """Calculate ideal average precision at k."""
    # Only keep items in predicted that are in actual
    predicted = [p for p in predicted if p in actual]
    return average_precision_at_k(actual, predicted, k)


def ideal_mean_average_precision_at_k(map_true: dict, map_pred: dict, k: int = 12) -> float:
    """Calculate ideal mean average precision at k."""
    logger.info(f"Evaluating ranking")
    apks = []
    for c_id, gt in map_true.items():
        pred = map_pred.get(c_id, [])
        apks.append(ideal_average_precision_at_k(gt, pred[:k]))
    logger.info(f"Ideal mean average precision at k: {np.mean(apks)}")
    return np.mean(apks)
