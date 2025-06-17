import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.sampling.popularity import PopularityBasedSampler


@pytest.mark.parametrize(
    "unique_customer_week_pairs, popular_items, weekly_stats, expected_result",
    [
        # Basic case
        (
            pd.DataFrame({"customer_id": [1, 1, 2], "week_num": [1, 2, 1]}),
            pd.DataFrame({"week_num": [0, 0, 1, 1, 2, 2], "article_id": [1, 2, 2, 3, 1, 3]}),
            pd.DataFrame(
                {
                    "week_num": [0, 0, 1, 1],
                    "article_id": [1, 2, 2, 3],
                    "avg_price": [10.0, 20.0, 30.0, 40.0],
                    "mode_sales_channel_id": [1, 2, 1, 2],
                }
            ),
            pd.DataFrame(
                {
                    "customer_id": [1, 1, 1, 1, 2, 2],
                    "week_num": [1, 1, 2, 2, 1, 1],
                    "article_id": [1, 2, 2, 3, 1, 2],
                    "price": [10.0, 20.0, 30.0, 40.0, 10.0, 20.0],
                    "sales_channel_id": [1, 2, 1, 2, 1, 2],
                    "source": ["popularity"] * 6,
                }
            ),
        ),
        # Only one customer
        (
            pd.DataFrame({"customer_id": [1], "week_num": [1]}),
            pd.DataFrame({"week_num": [0, 0, 1, 1, 2, 2], "article_id": [1, 2, 2, 3, 1, 3]}),
            pd.DataFrame(
                {
                    "week_num": [0, 0, 1, 1, 2, 2],
                    "article_id": [1, 2, 2, 3, 1, 3],
                    "avg_price": [10.0, 20.0, 30.0, 40.0, 10.0, 20.0],
                    "mode_sales_channel_id": [2, 1, 2, 1, 2, 1],
                }
            ),
            pd.DataFrame(
                {
                    "customer_id": [1, 1],
                    "week_num": [1, 1],
                    "article_id": [1, 2],
                    "price": [10.0, 20.0],
                    "sales_channel_id": [2, 1],
                    "source": ["popularity"] * 2,
                }
            ),
        ),
    ],
)
def test_popularity_based_sampler(unique_customer_week_pairs, popular_items, weekly_stats, expected_result):
    """Test overall functionality of PopularityBasedSampler class."""
    popularity_sampler = PopularityBasedSampler()
    result = popularity_sampler.generate(
        unique_customer_week_pairs=unique_customer_week_pairs,
        popular_items=popular_items,
        weekly_stats=weekly_stats,
    )
    assert_frame_equal(result, expected_result)


@pytest.mark.parametrize(
    "customer_week_pairs, popular_items, expected_result",
    [
        # Basic case
        (
            pd.DataFrame({"customer_id": [1, 1, 2], "week_num": [1, 2, 1]}),
            pd.DataFrame({"week_num": [0, 0, 1, 1, 2, 2], "article_id": [1, 2, 2, 3, 1, 3]}),
            pd.DataFrame(
                {"customer_id": [1, 1, 1, 1, 2, 2], "week_num": [1, 1, 2, 2, 1, 1], "article_id": [1, 2, 2, 3, 1, 2]}
            ),
        ),
        # Only one customer
        (
            pd.DataFrame({"customer_id": [1], "week_num": [1]}),
            pd.DataFrame({"week_num": [0, 0, 1, 1, 2, 2], "article_id": [1, 2, 2, 3, 1, 3]}),
            pd.DataFrame({"customer_id": [1, 1], "week_num": [1, 1], "article_id": [1, 2]}),
        ),
    ],
)
def test_popularity_based_sampler_merge_with_popular_items(customer_week_pairs, popular_items, expected_result):
    result = PopularityBasedSampler._merge_with_popular_items(customer_week_pairs, popular_items)
    assert_frame_equal(result, expected_result)


@pytest.mark.parametrize(
    "samples, weekly_stats, expected_result",
    [
        (
            pd.DataFrame(
                {"customer_id": [1, 1, 1, 1, 2, 2], "week_num": [1, 1, 2, 2, 1, 1], "article_id": [1, 2, 2, 3, 1, 2]}
            ),
            pd.DataFrame(
                {
                    "week_num": [0, 0, 1, 1],
                    "article_id": [1, 2, 2, 3],
                    "avg_price": [10.0, 20.0, 30.0, 40.0],
                    "mode_sales_channel_id": [1, 2, 1, 2],
                }
            ),
            pd.DataFrame(
                {
                    "customer_id": [1, 1, 1, 1, 2, 2],
                    "week_num": [1, 1, 2, 2, 1, 1],
                    "article_id": [1, 2, 2, 3, 1, 2],
                    "price": [10.0, 20.0, 30.0, 40.0, 10.0, 20.0],
                    "sales_channel_id": [1, 2, 1, 2, 1, 2],
                }
            ),
        ),
    ],
)
def test_popularity_based_sampler_add_item_metadata(samples, weekly_stats, expected_result):
    result = PopularityBasedSampler._add_item_metadata(samples, weekly_stats)
    assert_frame_equal(result, expected_result)
