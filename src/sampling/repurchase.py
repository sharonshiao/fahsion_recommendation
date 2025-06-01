import logging
from typing import List, Optional

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
