import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.sampling.popularity import PopularityBasedSampler
from src.sampling.repurchase import RepurchaseSampleGenerator
from src.sampling.utils import (
    calculate_article_weekly_stats,
    calculate_weekly_popular_items,
)

logger = logging.getLogger(__name__)

WEEK_NUM_VALID = 103
WEEK_NUM_TEST = 104


class NegativeSamplingManager:
    """Manager class for generating and combining negative samples from multiple strategies.

    This class orchestrates the generation of negative samples from different strategies
    (popularity-based, repurchase-based) and combines them into a single dataset for training
    or inference. It provides a unified interface for negative sampling operations.
    """

    def __init__(self, sampling_strategies: Dict[str, Dict[str, Any]]):
        """Initialize the sampling manager with strategies configuration.

        Args:
            sampling_strategies: Dictionary mapping strategy names to their configurations
                Example: {'popularity': {'top_k_items': 12}, 'repurchase': {"strategy": "last_k_items", "k": 12}}
        """
        self.sampling_strategies = sampling_strategies
        self.required_columns = ["customer_id", "week_num", "article_id", "price", "sales_channel_id"]
        self.samplers = {
            "popularity": PopularityBasedSampler(required_columns=self.required_columns),
            "repurchase": RepurchaseSampleGenerator(
                required_columns=self.required_columns,
                strategy=sampling_strategies["repurchase"]["strategy"],
                k=sampling_strategies["repurchase"]["k"],
            ),
        }
        logger.info(f"Initialized NegativeSamplingManager with strategies: {list(sampling_strategies.keys())}")

    def _calculate_popular_items(
        self, transactions: pd.DataFrame, week_num_start: int, week_num_end: int, top_k: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate popular items and weekly article statistics.

        Args:
            transactions: DataFrame with transaction data
            week_num_start: Start week number for the candidates
            week_num_end: End week number for the candidates
            top_k: Number of top items to include

        Returns:
            Tuple of (popular_items, weekly_article_stats)
        """
        logger.info(f"Calculating popular items with top_k={top_k}")

        # Use query string to filter transactions efficiently
        week_filter = f"week_num >= {week_num_start - 1} and week_num <= {week_num_end - 1}"
        filtered_transactions = transactions.query(week_filter)

        # Calculate popular items
        popular_items = calculate_weekly_popular_items(df=filtered_transactions, top_k=top_k)

        # Calculate weekly article statistics
        weekly_stats = calculate_article_weekly_stats(df=filtered_transactions)

        logger.debug(f"Generated {len(popular_items)} popular item entries")
        return popular_items, weekly_stats

    def _get_default_prediction(self, popular_items: pd.DataFrame, week_num: int) -> Optional[np.ndarray]:
        """Get default prediction based on popular items from a specific week. The number of items in the default
        prediction is the same as the number of items in the popular items for the given week_num.

        Args:
            popular_items: DataFrame with popular items
            week_num: Week number to use for default prediction

        Returns:
            Array of article IDs or None
        """
        # Filter for the specific week
        week_items = popular_items[popular_items["week_num"] == week_num]

        if len(week_items) == 0:
            logger.warning(f"No popular items found for week {week_num}")
            return None

        # Extract article IDs as an array
        default_prediction = week_items["article_id"].to_numpy()
        logger.debug(f"Created default prediction with {len(default_prediction)} items")

        return default_prediction

    def _generate_popularity_samples(
        self, unique_customer_week_pairs: pd.DataFrame, popular_items: pd.DataFrame, weekly_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate negative samples using popularity-based strategy.

        Args:
            unique_customer_week_pairs: DataFrame with unique customer-week pairs
            popular_items: DataFrame with popular items by week
            weekly_stats: DataFrame with weekly article statistics

        Returns:
            DataFrame with negative samples
        """
        if "popularity" not in self.sampling_strategies:
            logger.debug("Popularity strategy not enabled")
            return pd.DataFrame()

        logger.info("Generating popularity-based negative samples")
        sampler = self.samplers["popularity"]

        samples = sampler.generate(
            unique_customer_week_pairs=unique_customer_week_pairs,
            popular_items=popular_items,
            weekly_stats=weekly_stats,
        )

        logger.debug(f"Generated {len(samples)} popularity-based samples")
        logger.debug(f"Columns of samples: {samples.columns}")
        return samples

    def _generate_repurchase_samples(
        self,
        transactions: pd.DataFrame,
        week_num_start: int,
        week_num_end: int,
        customers_ids: Optional[List[int]] = None,
        default_future_week: int = 103,
    ) -> pd.DataFrame:
        """Generate negative samples using repurchase-based strategy.

        Args:
            transactions: DataFrame with transaction data
            week_num_start: Start week number
            week_num_end: End week number
            customers_ids: Optional list of customer IDs to filter by
            default_future_week: Week number for future weeks

        Returns:
            DataFrame with negative samples
        """
        if "repurchase" not in self.sampling_strategies:
            logger.debug("Repurchase strategy not enabled")
            return pd.DataFrame()

        logger.info("Generating repurchase-based negative samples")
        sampler = self.samplers["repurchase"]

        samples = sampler.generate(
            transactions=transactions,
            week_num_start=week_num_start,
            week_num_end=week_num_end,
            customers_ids=customers_ids,
            default_future_week=default_future_week,
        )

        logger.debug(f"Generated {len(samples)} repurchase-based samples")
        logger.debug(f"Columns of samples: {samples.columns}")
        return samples

    def combine_samples(
        self,
        transactions: pd.DataFrame,
        list_candidates: List[pd.DataFrame],
        sample_type: str = "train",
        restrict_negative_samples: bool = True,
    ) -> pd.DataFrame:
        """Combine negative samples from different sources and remove duplicates.

        Args:
            transactions: Original transaction data
            list_candidates: List of DataFrames with negative samples
            sample_type: 'train' or 'inference'
            restrict_negative_samples: If True, we drop negative samples that are in the observed transactions

        Returns:
            Combined and deduplicated negative samples
        """
        logger.info("Combining negative samples from multiple sources")

        # Skip if no candidates
        if not list_candidates or all(len(df) == 0 for df in list_candidates):
            logger.warning("No negative samples to combine")
            return pd.DataFrame()

        # Concatenate candidates
        negative_samples = pd.concat(list_candidates, ignore_index=True)
        logger.debug(f"Combined {len(negative_samples)} samples before deduplication")

        # Remove duplicates across sources
        negative_samples.drop_duplicates(inplace=True)

        # For training data, remove samples that exist in actual transactions
        if sample_type == "train" and restrict_negative_samples:
            logger.debug("Removing samples that exist in original transactions")

            # Key columns for matching
            key_cols = ["customer_id", "week_num", "article_id"]

            # Merge with indicator
            negative_samples = negative_samples.merge(transactions[key_cols], on=key_cols, how="left", indicator=True)

            # Keep only samples not in transactions
            negative_samples = negative_samples[negative_samples["_merge"] == "left_only"].drop(columns=["_merge"])

            # Remove rows with missing article_id
            negative_samples.dropna(subset=["article_id"], inplace=True)

        # Final deduplication by customer, week, article
        negative_samples.drop_duplicates(subset=["customer_id", "week_num", "article_id"], inplace=True)
        negative_samples.reset_index(drop=True, inplace=True)

        logger.debug(f"Final negative sample count: {len(negative_samples)}")
        return negative_samples

    def generate_samples(
        self,
        transactions: pd.DataFrame,
        unique_customer_week_pairs: pd.DataFrame,
        week_num_start: int,
        week_num_end: int,
        sample_type: str = "train",
        customers_ids: Optional[List[int]] = None,
        restrict_negative_samples: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], pd.DataFrame]:
        """Generate negative samples using configured strategies.

        This method orchestrates the negative sample generation process:
        1. Calculate popular items and weekly stats
        2. Generate samples from each enabled strategy
        3. Combine and deduplicate samples
        4. Create default predictions for inference

        Args:
            transactions: DataFrame with transaction history
            unique_customer_week_pairs: DataFrame with unique customer-week pairs
            week_num_start: Start week number
            week_num_end: End week number
            sample_type: 'train' or 'inference'
            customers_ids: Optional list of customer IDs to filter
            restrict_negative_samples: If True, we drop negative samples that are in the observed transactions

        Returns:
            Tuple of (negative_samples, default_prediction, popular_items)
        """
        logger.info(f"Generating negative samples for {sample_type}")
        logger.debug(f"Active strategies: {list(self.sampling_strategies.keys())}")

        # Calculate popular items and weekly stats
        top_k = self.sampling_strategies.get("popularity", {}).get("top_k_items", 12)
        popular_items, weekly_stats = self._calculate_popular_items(
            transactions=transactions, week_num_start=week_num_start, week_num_end=week_num_end, top_k=top_k
        )

        # Default prediction for inference
        default_prediction = None
        if sample_type != "train":
            default_prediction = self._get_default_prediction(
                popular_items=popular_items, week_num=week_num_start - 1  # Previous week
            )

        # List to collect samples from different strategies
        all_samples = []

        # Generate popularity-based samples
        if "popularity" in self.sampling_strategies:
            popularity_samples = self._generate_popularity_samples(
                unique_customer_week_pairs=unique_customer_week_pairs,
                popular_items=popular_items,
                weekly_stats=weekly_stats,
            )
            if len(popularity_samples) > 0:
                all_samples.append(popularity_samples)

        # Generate repurchase-based samples
        if "repurchase" in self.sampling_strategies:
            if sample_type == "test":
                default_future_week = WEEK_NUM_TEST
            else:
                default_future_week = WEEK_NUM_VALID

            repurchase_samples = self._generate_repurchase_samples(
                transactions=transactions,
                week_num_start=week_num_start,
                week_num_end=week_num_end,
                customers_ids=customers_ids,
                default_future_week=default_future_week,
            )
            if len(repurchase_samples) > 0:
                all_samples.append(repurchase_samples)

        # Combine samples from all strategies
        negative_samples = self.combine_samples(
            transactions=transactions,
            list_candidates=all_samples,
            sample_type=sample_type,
            restrict_negative_samples=restrict_negative_samples,
        )

        return negative_samples, default_prediction, popular_items
