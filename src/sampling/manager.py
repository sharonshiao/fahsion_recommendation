import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RepurchaseSampleGenerator:
    """Class to generate negative samples from previous purchases.

    This class handles the generation of negative samples by projecting previous purchases
    into future weeks. It breaks down the process into smaller, maintainable steps and
    allows for more flexibility in customizing the generation process.
    """

    def __init__(
        self,
        required_columns: List[str] = ["customer_id", "week_num", "article_id", "price", "sales_channel_id"],
    ):
        """Initialize the generator with configuration.

        Args:
            required_columns: List of required columns in transaction data
        """
        self.required_columns = required_columns

    @staticmethod
    def _validate_input(transactions: pd.DataFrame, required_columns: List[str]) -> None:
        """Validate input data has required columns.

        Args:
            transactions: Input transaction DataFrame
            required_columns: List of required columns

        Raises:
            ValueError: If required columns are missing
        """
        missing_cols = set(required_columns) - set(transactions.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    @staticmethod
    def _filter_customers(df: pd.DataFrame, customers_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Filter transactions by customer IDs if provided.

        Args:
            df: Input DataFrame
            customers_ids: Optional list of customer IDs to filter by

        Returns:
            Filtered DataFrame
        """
        if customers_ids is not None:
            logger.debug(f"Filtering transactions by customer_ids: {customers_ids}")
            logger.debug("Filtering transactions by customer_ids")
            return df[df["customer_id"].isin(customers_ids)]
        else:
            logger.debug("No customer_ids provided, returning all transactions")
        logger.debug(f"Shape of filtered transactions: {df.shape}")
        return df

    @staticmethod
    def _create_week_mapping(df: pd.DataFrame, default_future_week: int) -> Dict[Tuple[int, int], int]:
        """Create mapping from current weeks to future weeks for each customer.

        Args:
            df: Input DataFrame
            default_future_week: Week number to use for last purchase

        Returns:
            Dictionary mapping (customer_id, week_num) to next week
        """
        logger.debug("Creating week mapping")
        # Create a sorted customer-week DataFrame
        customer_weeks = df[["customer_id", "week_num"]].drop_duplicates().sort_values(["customer_id", "week_num"])

        # Create next week mapping
        customer_weeks["next_week"] = customer_weeks.groupby("customer_id")["week_num"].shift(-1)
        customer_weeks["next_week"] = customer_weeks["next_week"].fillna(default_future_week)

        # Create and return the mapping dictionary
        logger.debug(f"Shape of customer_weeks: {customer_weeks.shape}")
        return customer_weeks.set_index(["customer_id", "week_num"])["next_week"].to_dict()

    @staticmethod
    def _apply_week_mapping(
        df: pd.DataFrame,
        required_columns: List[str],
        week_mapping: Dict[Tuple[int, int], int],
        week_num_start: int,
        week_num_end: int,
    ) -> pd.DataFrame:
        """Apply week mapping to transactions and filter by week range.

        Args:
            df: Input DataFrame
            required_columns: List of required columns
            week_mapping: Dictionary mapping (customer_id, week_num) to next week
            week_num_start: Start week number for filtering
            week_num_end: End week number for filtering

        Returns:
            DataFrame with mapped weeks and filtered by range
        """
        logger.debug("Applying week mapping")
        # Apply mapping
        prev_transactions = df[required_columns].copy()
        prev_transactions["week_num_new"] = prev_transactions.apply(
            lambda x: week_mapping.get((x["customer_id"], x["week_num"])), axis=1
        )

        # Filter by week range
        mask = (prev_transactions["week_num_new"] >= week_num_start) & (
            prev_transactions["week_num_new"] <= week_num_end
        )
        prev_transactions = prev_transactions[mask]

        # Update week_num and clean up
        prev_transactions["week_num"] = prev_transactions["week_num_new"].astype(int)
        prev_transactions.drop("week_num_new", axis=1, inplace=True)

        logger.debug(f"Shape of prev_transactions after applying week mapping: {prev_transactions.shape}")
        return prev_transactions

    def generate(
        self,
        transactions: pd.DataFrame,
        week_num_start: int,
        week_num_end: int,
        customers_ids: Optional[List[int]] = None,
        default_future_week: int = 103,
    ) -> pd.DataFrame:
        """Generate negative samples from previous purchases.

        This method orchestrates the negative sample generation process by:
        1. Validating input data
        2. Filtering by customers if needed
        3. Creating week mappings
        4. Applying mappings and filtering by week range
        5. Adding source information

        Args:
            transactions: DataFrame containing transaction data
            week_num_start: Start week number for prediction
            week_num_end: End week number for prediction
            customers_ids: Optional list of customer IDs to filter by
            default_future_week: Week number for future weeks

        Returns:
            DataFrame containing negative samples for weeks between week_num_start and week_num_end.
        """
        logger.info("Preparing negative samples by previous purchases")

        # Validate and prepare input
        self._validate_input(transactions, self.required_columns)
        df = self._filter_customers(transactions.copy(), customers_ids)

        # Create week mapping
        week_mapping = self._create_week_mapping(df, default_future_week)

        # Apply mapping and filter
        result = self._apply_week_mapping(df, self.required_columns, week_mapping, week_num_start, week_num_end)

        # Add source column
        result["source"] = "repurchase"

        logger.debug(f"Generated {len(result)} negative samples")
        logger.debug(f"Columns of result: {result.columns}")
        return result


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
                Example: {'popularity': {'top_k_items': 12}, 'repurchase': {}}
        """
        self.sampling_strategies = sampling_strategies
        self.required_columns = ["customer_id", "week_num", "article_id", "price", "sales_channel_id"]
        self.samplers = {
            "popularity": PopularityBasedSampler(required_columns=self.required_columns),
            "repurchase": RepurchaseSampleGenerator(required_columns=self.required_columns),
        }
        logger.info(f"Initialized NegativeSamplingManager with strategies: {list(sampling_strategies.keys())}")

    def _calculate_popular_items(
        self, transactions: pd.DataFrame, week_num_start: int, week_num_end: int, top_k: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate popular items and weekly article statistics.

        Args:
            transactions: DataFrame with transaction data
            week_num_start: Start week number
            week_num_end: End week number
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
        """Get default prediction based on popular items from a specific week.

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
        self, transactions: pd.DataFrame, list_candidates: List[pd.DataFrame], sample_type: str = "train"
    ) -> pd.DataFrame:
        """Combine negative samples from different sources and remove duplicates.

        Args:
            transactions: Original transaction data
            list_candidates: List of DataFrames with negative samples
            sample_type: 'train' or 'inference'

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
        if sample_type == "train":
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
            repurchase_samples = self._generate_repurchase_samples(
                transactions=transactions,
                week_num_start=week_num_start,
                week_num_end=week_num_end,
                customers_ids=customers_ids,
            )
            if len(repurchase_samples) > 0:
                all_samples.append(repurchase_samples)

        # Combine samples from all strategies
        negative_samples = self.combine_samples(
            transactions=transactions, list_candidates=all_samples, sample_type=sample_type
        )

        return negative_samples, default_prediction, popular_items
