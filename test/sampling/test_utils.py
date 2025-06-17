import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.sampling.utils import (
    calculate_article_weekly_stats,
    calculate_rolling_popular_items,
    calculate_weekly_popular_items,
)


@pytest.fixture
def sample_transaction_data_with_counts():
    """Fixture providing sample transaction data for testing."""
    purchases = {
        "1": {1: 3, 2: 2, 3: 1},
        "2": {1: 0, 2: 3, 3: 1},
        "3": {1: 1, 2: 1, 3: 0},
        "4": {1: 0, 2: 0, 3: 2},
    }
    rows = []
    for week_num, articles in purchases.items():
        for article_id, quantity in articles.items():
            # Add a row for each purchase (quantity times)
            for _ in range(quantity):
                rows.append({"week_num": int(week_num), "article_id": article_id})

    return pd.DataFrame(rows)


@pytest.fixture
def sample_transaction_data_with_quantity():
    """Fixture providing sample transaction data for testing."""
    purchases = {
        "1": {1: 3, 2: 2, 3: 1},
        "2": {1: 0, 2: 3, 3: 1},
        "3": {1: 1, 2: 1, 3: 0},
        "4": {1: 0, 2: 0, 3: 2},
    }
    data = []
    for week_num, articles in purchases.items():
        for article_id, quantity in articles.items():
            data.append({"week_num": int(week_num), "article_id": article_id, "quantity": quantity})
    return pd.DataFrame(data)


@pytest.fixture
def weekly_transaction_data():
    """Fixture providing sample transaction data for weekly stats testing."""
    return pd.DataFrame(
        {
            "week_num": [1, 1, 1, 1, 2, 2, 2, 2],
            "article_id": [1, 1, 2, 3, 1, 2, 2, 3],
            "t_dat": pd.concat(
                [pd.Series([pd.to_datetime("2020-01-01")] * 4), pd.Series([pd.to_datetime("2020-01-08")] * 4)]
            ),
            "price": [10.0, 15.0, 20.0, 30.0, 15.0, 25.0, 35.0, 35.0],
            "sales_channel_id": [1, 1, 2, 1, 1, 2, 1, 1],
        }
    )


@pytest.fixture
def expected_weekly_stats():
    """Fixture providing expected weekly stats for testing."""
    return pd.DataFrame(
        {
            "week_num": [1, 1, 1, 2, 2, 2],
            "article_id": [1, 2, 3, 1, 2, 3],
            "count": [2, 1, 1, 1, 2, 1],
            "avg_price": [12.5, 20.0, 30.0, 15.0, 30.0, 35.0],
            "mode_sales_channel_id": [1, 2, 1, 1, 1, 1],
        }
    )


@pytest.mark.parametrize(
    "window_size, top_k, expected_result",
    [
        # Basic case: window_size = 1, top_k = 2
        (
            1,
            2,
            # Expected results for each week:
            # Week 1: article 1 (3 sales), article 2 (2 sales), article 3 (1 sale)
            # Week 2: article 1 (0 sales), article 2 (3 sales), article 3 (1 sale)
            # Week 3: article 1 (1 sale), article 2 (1 sale), article 3 (0 sales)
            # Week 4: article 3 (2 sales), article 1 (0 sales), article 2 (0 sales)
            pd.DataFrame(
                {
                    "week_num": [1, 1, 2, 2, 3, 3, 4, 4],
                    "article_id": [1, 2, 2, 3, 1, 2, 3, 1],
                }
            ),
        ),
        # Window_size = 2, top_k = 2
        (
            2,
            2,
            # Expected results for each week:
            # Week 0-1: article 1 (3 sales), article 2 (2 sales), article 3 (1 sales)
            # Week 1-2: article 1 (3 sales), article 2 (5 sales), article 3 (2 sales)
            # Week 2-3: article 1 (1 sales), article 2 (4 sales), article 3 (1 sales)
            # Week 3-4: article 1 (1 sales), article 2 (1 sales) article 3 (2 sales)
            pd.DataFrame(
                {
                    "week_num": [1, 1, 2, 2, 3, 3, 4, 4],
                    "article_id": [1, 2, 2, 1, 2, 1, 3, 1],
                }
            ),
        ),
        # Window_size = 2, top_k = 3
        (
            2,
            3,
            pd.DataFrame(
                {
                    "week_num": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                    "article_id": [1, 2, 3, 2, 1, 3, 2, 1, 3, 3, 1, 2],
                }
            ),
        ),
    ],
)
def test_calculate_rolling_popular_items_basic(
    sample_transaction_data_with_counts, window_size, top_k, expected_result
):
    """Test basic functionality of rolling popular items calculation."""
    result = calculate_rolling_popular_items(
        df=sample_transaction_data_with_counts,
        window_size=window_size,
        top_k=top_k,
    )

    assert_frame_equal(result, expected_result)


@pytest.mark.parametrize(
    "window_size, top_k, expected_result",
    [
        # Window_size = 1, top_k = 2
        (
            1,
            2,
            pd.DataFrame(
                {
                    "week_num": [1, 1, 2, 2, 3, 3, 4, 4],
                    "article_id": [1, 2, 2, 3, 1, 2, 3, 1],
                }
            ),
        ),
        # Window_size = 2, top_k = 2
        (
            2,
            2,
            pd.DataFrame(
                {
                    "week_num": [1, 1, 2, 2, 3, 3, 4, 4],
                    "article_id": [1, 2, 2, 1, 2, 1, 3, 1],
                }
            ),
        ),
        # Window_size = 2, top_k = 3
        (
            2,
            3,
            pd.DataFrame(
                {
                    "week_num": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                    "article_id": [1, 2, 3, 2, 1, 3, 2, 1, 3, 3, 1, 2],
                }
            ),
        ),
    ],
)
def test_calculate_rolling_popular_items_with_quantity(
    sample_transaction_data_with_quantity, window_size, top_k, expected_result
):
    """Test calculation using quantity column instead of counts."""
    result = calculate_rolling_popular_items(
        df=sample_transaction_data_with_quantity,
        window_size=window_size,
        top_k=top_k,
        value_col="quantity",
    )
    assert_frame_equal(result, expected_result)


def test_calculate_rolling_popular_items_missing_weeks():
    """Test handling of missing weeks in the data."""
    df = pd.DataFrame(
        {
            "week_num": [1, 1, 3, 3, 5],  # Week 2 and 4 missing
            "article_id": [1, 2, 3, 1, 2],
        }
    )

    result = calculate_rolling_popular_items(df, window_size=2, top_k=2)

    # Should have results for all weeks 1-5
    assert set(result["week_num"].unique()) == {1, 2, 3, 4, 5}


def test_calculate_rolling_popular_items_edge_cases():
    """Test edge cases like empty data or small windows."""
    # Empty DataFrame
    empty_df = pd.DataFrame(columns=["week_num", "article_id"])
    with pytest.raises(ValueError):
        calculate_rolling_popular_items(empty_df, window_size=1, top_k=1)

    # Single row
    single_row_df = pd.DataFrame({"week_num": [1], "article_id": [1]})
    result = calculate_rolling_popular_items(single_row_df, window_size=1, top_k=1)
    assert len(result) == 1
    assert result.iloc[0]["article_id"] == 1

    # Window size larger than data range
    result = calculate_rolling_popular_items(single_row_df, window_size=5, top_k=1)
    assert len(result) == 1
    assert result.iloc[0]["article_id"] == 1


def test_calculate_weekly_popular_items(weekly_transaction_data):
    """Test weekly popular items calculation."""
    result = calculate_weekly_popular_items(
        df=weekly_transaction_data,
        top_k=2,  # Get top 2 items per week
    )

    # Expected results:
    # Week 1: article 1 (2 sales, rank 1), article 2 (1 sale, rank 2)
    # Week 2: article 2 (2 sales, rank 1), article 1
    expected = pd.DataFrame(
        {
            "week_num": [1, 1, 2, 2],
            "article_id": [1, 2, 2, 1],
            "bestseller_rank": [1.0, 2.0, 1.0, 2.0],
        }
    )
    print(result)
    print(expected)

    # Sort both dataframes to ensure consistent comparison
    result = result.sort_values(["week_num", "article_id"]).reset_index(drop=True)
    expected = expected.sort_values(["week_num", "article_id"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(result, expected)


def test_calculate_weekly_popular_items_custom_columns(weekly_transaction_data):
    """Test weekly popular items calculation with custom column names."""
    # Rename columns in test data
    df = weekly_transaction_data.rename(columns={"week_num": "week", "article_id": "item_id"})

    result = calculate_weekly_popular_items(
        df=df,
        top_k=2,
        week_col="week",
        article_col="item_id",
    )

    assert "week" in result.columns
    assert "item_id" in result.columns
    assert "bestseller_rank" in result.columns


def test_calculate_article_weekly_stats(weekly_transaction_data, expected_weekly_stats):
    """Test article weekly statistics calculation."""
    result = calculate_article_weekly_stats(df=weekly_transaction_data)
    result = result.sort_values(["week_num", "article_id"]).reset_index(drop=True)

    # Verify the structure and content of the results
    assert set(result.columns) == {"week_num", "article_id", "count", "avg_price", "mode_sales_channel_id"}

    assert_frame_equal(result, expected_weekly_stats)


def test_calculate_article_weekly_stats_custom_columns(weekly_transaction_data):
    """Test article weekly stats calculation with custom column names."""
    # Rename columns in test data
    df = weekly_transaction_data.rename(
        columns={
            "week_num": "week",
            "article_id": "item_id",
            "t_dat": "date",
            "price": "unit_price",
            "sales_channel_id": "channel",
        }
    )

    result = calculate_article_weekly_stats(
        df=df, week_col="week", article_col="item_id", date_col="date", price_col="unit_price", channel_col="channel"
    )

    assert set(result.columns) == {"week", "item_id", "count", "avg_price", "mode_sales_channel_id"}


def test_calculate_article_weekly_stats_empty_data():
    """Test article weekly stats calculation with empty data."""
    empty_df = pd.DataFrame(columns=["week_num", "article_id", "t_dat", "price", "sales_channel_id"])
    result = calculate_article_weekly_stats(df=empty_df)
    assert len(result) == 0
    assert set(result.columns) == {"week_num", "article_id", "count", "avg_price", "mode_sales_channel_id"}
