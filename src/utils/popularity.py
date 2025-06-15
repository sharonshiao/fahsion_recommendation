import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_rolling_popular_items(
    df: pd.DataFrame,
    window_size: int,
    top_k: int,
    week_col: str = "week_num",
    article_col: str = "article_id",
    value_col: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate top k popular items over a rolling window.

    This function efficiently calculates the most popular items for each week
    based on a rolling window of previous weeks. It uses pandas groupby and
    rolling operations for optimal performance.

    Args:
        df: DataFrame containing transaction data
        window_size: Number of weeks to look back (t)
        top_k: Number of top items to return (k)
        week_col: Name of the column containing week numbers
        article_col: Name of the column containing article IDs
        value_col: Optional column to use for aggregation (e.g. quantity).
                  If None, counts occurrences.

    Returns:
        DataFrame with columns [week_num, article_id] containing the top k
        popular items for each week based on the previous t weeks.

    Example:
        >>> df = pd.DataFrame({
        ...     'week_num': [1, 1, 1, 2, 2, 3],
        ...     'article_id': [1, 2, 1, 2, 3, 1],
        ... })
        >>> calculate_rolling_popular_items(df, window_size=2, top_k=2)
           week_num  article_id
        0        1          1
        1        1          2
        2        2          1
        3        2          2
        4        3          2
        5        3          3
    """
    logger.info(f"Calculating top {top_k} popular items with {window_size} week window")

    # Sort by week to ensure correct rolling window
    df = df.sort_values(week_col)

    # Get unique weeks for output
    unique_weeks = df[week_col].unique()
    min_week = unique_weeks.min()
    max_week = unique_weeks.max()

    # Create a complete range of weeks to handle missing weeks
    all_weeks = np.arange(min_week, max_week + 1)

    # Group by week and article, count occurrences or sum values
    if value_col is None:
        weekly_counts = df.groupby([week_col, article_col]).size().reset_index(name="count")
    else:
        weekly_counts = df.groupby([week_col, article_col])[value_col].sum().reset_index()

    # Pivot to get articles as columns
    pivot_df = weekly_counts.pivot(
        index=week_col, columns=article_col, values="count" if value_col is None else value_col
    ).fillna(0)

    # Ensure all weeks are present
    pivot_df = pivot_df.reindex(all_weeks, fill_value=0)

    # Calculate rolling sum for each article
    rolling_sums = pivot_df.rolling(window=window_size, min_periods=1).sum()

    # Get top k articles for each week
    top_articles = []
    for week in all_weeks:
        # Get the row for current week
        week_data = rolling_sums.loc[week]

        # Sort articles by count/value and get top k
        top_k_articles = week_data.nlargest(top_k).index.tolist()

        # Add to results
        top_articles.extend([(week, article_id) for article_id in top_k_articles])

    # Create output DataFrame
    result = pd.DataFrame(top_articles, columns=[week_col, article_col])

    logger.debug(f"Generated {len(result)} popular item entries")
    return result


def calculate_weekly_popular_items(
    df: pd.DataFrame,
    top_k: int,
    week_col: str = "week_num",
    article_col: str = "article_id",
) -> pd.DataFrame:
    """Calculate the most popular items for each week.

    Args:
        df: DataFrame containing transaction data
        top_k: Number of top items to return
        week_col: Name of the week column
        article_col: Name of the article column

    Returns:
        DataFrame with bestseller rank for each week-article combination
    """
    logger.info("Calculating weekly popular items")
    top_k_articles_by_week = (
        df.groupby(week_col)[article_col]
        .value_counts()
        .groupby(week_col)
        .rank(method="dense", ascending=False)
        .groupby(week_col)
        .head(top_k)
        .rename("bestseller_rank")
    ).reset_index()

    logger.debug(f"Shape of top_k_articles_by_week: {top_k_articles_by_week.shape}")
    return top_k_articles_by_week


def calculate_article_weekly_stats(
    df: pd.DataFrame,
    week_col: str = "week_num",
    article_col: str = "article_id",
    date_col: str = "t_dat",
    price_col: str = "price",
    channel_col: str = "sales_channel_id",
) -> pd.DataFrame:
    """Calculate weekly statistics for each article.

    Args:
        df: DataFrame containing transaction data
        week_col: Name of the week column
        article_col: Name of the article column
        date_col: Name of the date column
        price_col: Name of the price column
        channel_col: Name of the sales channel column

    Returns:
        DataFrame with weekly stats (count, avg_price, mode_channel) for each article
    """
    logger.info("Calculating article weekly stats")
    article_weekly_stats = (
        df.groupby([week_col, article_col])
        .agg(
            count=(date_col, "count"),
            avg_price=(price_col, "mean"),
            mode_sales_channel_id=(channel_col, lambda x: x.mode()[0]),
        )
        .reset_index()
        .sort_values([week_col, "count"], ascending=[True, False])
    )

    logger.debug(f"Shape of article_weekly_stats: {article_weekly_stats.shape}")
    return article_weekly_stats


class NegativeSampleGenerator:
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
            logger.debug("Filtering transactions by customer_ids")
            return df[df["customer_id"].isin(customers_ids)]
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
        # Create a sorted customer-week DataFrame
        customer_weeks = df[["customer_id", "week_num"]].drop_duplicates().sort_values(["customer_id", "week_num"])

        # Create next week mapping
        customer_weeks["next_week"] = customer_weeks.groupby("customer_id")["week_num"].shift(-1)
        customer_weeks["next_week"] = customer_weeks["next_week"].fillna(default_future_week)

        # Create and return the mapping dictionary
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
            week_mapping: Dictionary mapping (customer_id, week_num) to next week
            week_num_start: Start week number for filtering
            week_num_end: End week number for filtering

        Returns:
            DataFrame with mapped weeks and filtered by range
        """
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
            DataFrame containing negative samples with projected future weeks
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
        return result


# FIX
# def prepare_negative_samples_by_previous_purchases(
#     transactions: pd.DataFrame,
#     week_num_start: int,
#     week_num_end: int,
#     customers_ids: Optional[List[int]] = None,
#     default_future_week: int = 103,
#     required_columns: List[str] = ["customer_id", "week_num", "article_id", "price", "sales_channel_id"],
# ) -> pd.DataFrame:
#     """Legacy wrapper for backward compatibility.

#     This function maintains backward compatibility while using the new class-based implementation.
#     """
#     generator = NegativeSampleGenerator(required_columns=required_columns)
#     return generator.generate(
#         transactions=transactions,
#         week_num_start=week_num_start,
#         week_num_end=week_num_end,
#         customers_ids=customers_ids,
#         default_future_week=default_future_week,
#     )
