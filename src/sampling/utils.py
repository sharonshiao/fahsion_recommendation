import logging
from typing import Optional

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

    Notes:
        - If a week has no sales, the week will be present in the output but will have all zeros. An item with no sales
          may be present in the output if all items in the week have no sales.
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
        .reset_index(name="bestseller_rank")
        # Sort by week, rank, article to ensure consistent ordering
        .sort_values([week_col, "bestseller_rank", article_col], ascending=[True, True, True])
        .groupby(week_col)
        # Note: we pick the top_k rows here so there can be items of the same ranks but not chosen because of having the
        # same rank
        .head(top_k)
    ).reset_index(drop=True)

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


def prepare_negative_samples_by_popularity(
    # Positive transactions only need to consists of customer_id and week_num
    positive_transactions: pd.DataFrame,
    candidates_popular: pd.DataFrame,
    weekly_articles_stats: pd.DataFrame = None,
):
    """Prepare negative samples for each row in positive samples by picking the most popular item in the same category
    in the previous week."""
    logger.info("Preparing negative samples by popularity")
    # Get unique (customer_id, week) pairs from positive transactions
    cols_transactions = ["customer_id", "week_num"]
    base = positive_transactions[cols_transactions].drop_duplicates().copy()
    base["prev_week_num"] = base["week_num"] - 1
    logger.debug(f"Number of unique (customer_id, week) pairs: {len(base)}")

    # Merge with candidates to get the most popular item in the previous week
    cols_candidates = [
        "week_num",
        "article_id",
    ]

    negative_transactions_popular = base.merge(
        candidates_popular[cols_candidates],
        left_on=["prev_week_num"],
        right_on=["week_num"],
        how="left",
        suffixes=("", "_candidate"),
    )

    negative_transactions_popular = negative_transactions_popular.drop(columns=["week_num_candidate"])

    # Join with weekly articles stats
    if weekly_articles_stats is not None:
        logger.debug("Merging with weekly articles stats")
        negative_transactions_popular = negative_transactions_popular.merge(
            weekly_articles_stats[["week_num", "article_id", "avg_price", "mode_sales_channel_id"]],
            left_on=["prev_week_num", "article_id"],
            right_on=["week_num", "article_id"],
            how="left",
            suffixes=("", "_article"),
        )
        negative_transactions_popular.rename(
            columns={
                "avg_price": "price",
                "mode_sales_channel_id": "sales_channel_id",
            },
            inplace=True,
        )
        negative_transactions_popular = negative_transactions_popular.drop(columns=["week_num_article"])
        logger.debug(
            f"Shape of negative_transactions_popular after merging with weekly articles stats: {negative_transactions_popular.shape}"
        )

    negative_transactions_popular = negative_transactions_popular.drop(columns=["prev_week_num"])
    negative_transactions_popular = negative_transactions_popular.rename(
        columns={
            "article_id_candidate": "article_id",
        }
    )
    logger.debug(f"Number of negative samples: {len(negative_transactions_popular)}")

    negative_transactions_popular["source"] = "popularity"

    return negative_transactions_popular
