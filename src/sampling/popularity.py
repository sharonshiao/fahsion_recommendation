import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class PopularityBasedSampler:
    """Class to generate negative samples based on popularity metrics.

    This class creates negative samples by selecting popular items from previous weeks
    for each customer. It provides a modular approach to popularity-based sampling.
    """

    def __init__(
        self,
        required_columns: List[str] = ["customer_id", "week_num", "article_id", "price", "sales_channel_id"],
    ):
        """Initialize the sampler with configuration.

        Args:
            required_columns: List of required columns in the output
        """
        self.required_columns = required_columns

    @staticmethod
    def _merge_with_popular_items(
        customer_week_pairs: pd.DataFrame,
        popular_items: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge customer-week pairs with popular items from previous weeks.

        Args:
            customer_week_pairs: DataFrame with customer_id, week_num, prev_week_num
            popular_items: DataFrame with popular items by week

        Returns:
            DataFrame with customer_id, week_num, article_id from popular items
        """
        logger.debug("Merging customer-week pairs with popular items")
        cols_candidates = ["week_num", "article_id"]
        customer_week_pairs["prev_week_num"] = customer_week_pairs["week_num"] - 1

        result = customer_week_pairs.merge(
            popular_items[cols_candidates],
            left_on=["prev_week_num"],
            right_on=["week_num"],
            how="left",
            suffixes=("", "_candidate"),
        )

        # Clean up and rename columns
        result = result.drop(columns=["week_num_candidate"])
        result = result.drop(columns=["prev_week_num"])
        result = result.rename(columns={"article_id_candidate": "article_id"})
        logger.debug(f"Shape of result: {result.shape}")

        return result

    @staticmethod
    def _add_item_metadata(
        samples: pd.DataFrame,
        weekly_stats: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add item metadata (price, sales channel) from weekly statistics.

        Args:
            samples: DataFrame with customer_id, week_num, article_id
            weekly_stats: DataFrame with weekly article statistics

        Returns:
            DataFrame with added price and sales_channel_id
        """
        logger.debug("Adding item metadata from weekly statistics")

        if weekly_stats is not None:
            logger.debug("Adding item metadata from weekly statistics")
            # Calculate previous week for joining
            samples["prev_week_num"] = samples["week_num"] - 1

            # Join with weekly article stats
            result = samples.merge(
                weekly_stats[["week_num", "article_id", "avg_price", "mode_sales_channel_id"]],
                left_on=["prev_week_num", "article_id"],
                right_on=["week_num", "article_id"],
                how="left",
                suffixes=("", "_article"),
            )

            # Rename and clean up
            result = result.rename(
                columns={
                    "avg_price": "price",
                    "mode_sales_channel_id": "sales_channel_id",
                }
            )
            result = result.drop(columns=["week_num_article", "prev_week_num"])
            logger.debug(f"Shape of result after adding item metadata: {result.shape}")

            return result
        else:
            logger.debug("No weekly stats provided, skipping item metadata")
            return samples

    def generate(
        self,
        unique_customer_week_pairs: pd.DataFrame,
        popular_items: pd.DataFrame,
        weekly_stats: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate negative samples based on popular items.

        This method orchestrates the popularity-based sample generation by:
        1. Merging with popular items from previous weeks
        2. Adding item metadata if available
        3. Adding source information

        Args:
            unique_customer_week_pairs: DataFrame with unique customer-week pairs
            popular_items: DataFrame with popular items by week
            weekly_stats: Optional DataFrame with weekly article statistics

        Returns:
            DataFrame containing negative samples based on popular items
        """
        logger.info("Preparing negative samples by popularity")

        # Get unique customer-week pairs

        # Merge with popular items
        samples = self._merge_with_popular_items(unique_customer_week_pairs, popular_items)

        # Add item metadata if available
        if weekly_stats is not None:
            samples = self._add_item_metadata(samples, weekly_stats)

        # Add source column
        samples["source"] = "popularity"

        logger.debug(f"Generated {len(samples)} popularity-based samples")
        logger.debug(f"Columns of samples: {samples.columns}")
        return samples
