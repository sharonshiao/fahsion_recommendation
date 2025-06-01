import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.utils.popularity import (
    NegativeSampleGenerator,
    calculate_article_weekly_stats,
    calculate_rolling_popular_items,
    calculate_weekly_popular_items,
)


@pytest.fixture
def sample_transaction_data():
    """Fixture providing sample transaction data for testing."""
    return pd.DataFrame(
        {
            "week_num": [1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
            "article_id": [1, 2, 1, 3, 2, 3, 2, 1, 2, 3],
            "quantity": [1, 1, 1, 1, 2, 1, 1, 1, 1, 2],
        }
    )


@pytest.fixture
def weekly_transaction_data():
    """Fixture providing sample transaction data for weekly stats testing."""
    return pd.DataFrame(
        {
            "week_num": [1, 1, 1, 1, 2, 2, 2, 2],
            "article_id": [1, 1, 2, 3, 1, 2, 2, 3],
            "t_dat": pd.date_range("2020-01-01", periods=8),
            "price": [10.0, 10.0, 20.0, 30.0, 15.0, 25.0, 25.0, 35.0],
            "sales_channel_id": [1, 1, 2, 1, 1, 2, 2, 1],
        }
    )


# FIX
@pytest.fixture
def purchase_history_data():
    """Fixture providing sample purchase history data for testing."""
    return pd.DataFrame(
        {
            "customer_id": [1, 1, 1, 1, 2, 2, 3],
            "week_num": [0, 1, 2, 4, 1, 3, 2],
            "article_id": [100, 101, 102, 103, 201, 202, 301],
            "price": [5.0, 10.0, 20.0, 30.0, 15.0, 25.0, 35.0],
            "sales_channel_id": [2, 1, 2, 1, 1, 2, 1],
        }
    )


@pytest.fixture
def expected_negative_samples():
    """Fixture providing expected negative samples for testing."""
    return pd.DataFrame(
        {
            "customer_id": [1, 1, 1, 1, 2, 2, 3],
            "week_num": [1, 2, 4, 100, 3, 100, 100],
            "article_id": [100, 101, 102, 103, 201, 202, 301],
            "price": [5.0, 10.0, 20.0, 30.0, 15.0, 25.0, 35.0],
            "sales_channel_id": [2, 1, 2, 1, 1, 2, 1],
            "source": ["repurchase"] * 7,
        }
    )


# def test_calculate_rolling_popular_items_basic(sample_transaction_data):
#     """Test basic functionality of rolling popular items calculation."""
#     result = calculate_rolling_popular_items(
#         df=sample_transaction_data,
#         window_size=1,  # Only look at current week
#         top_k=2,  # Get top 2 items
#     )

#     # Expected results for each week:
#     # Week 1: article 1 (2 sales), article 2 (1 sale)
#     # Week 2: article 2 (2 sales), article 3 (1 sale)
#     # Week 3: article 1 (1 sale), article 2 (1 sale)
#     # Week 4: article 3 (2 sales)
#     expected = pd.DataFrame(
#         {
#             "week_num": [1, 1, 2, 2, 3, 3, 4, 4],
#             "article_id": [1, 2, 2, 3, 1, 2, 3, 2],
#         }
#     )
#     pd.testing.assert_frame_equal(
#         result.sort_values(["week_num", "article_id"]).reset_index(drop=True),
#         expected.sort_values(["week_num", "article_id"]).reset_index(drop=True),
#     )


# def test_calculate_rolling_popular_items_window(sample_transaction_data):
#     """Test rolling window functionality."""
#     result = calculate_rolling_popular_items(
#         df=sample_transaction_data,
#         window_size=2,  # Look at current and previous week
#         top_k=2,  # Get top 2 items
#     )

#     # Expected results for each week:
#     # Week 1: article 1 (2 sales), article 2 (1 sale)
#     # Week 2: article 2 (3 sales), article 1 (2 sales)
#     # Week 3: article 2 (3 sales), article 3 (2 sales)
#     # Week 4: article 2 (1 sale), article 3 (3 sales)
#     expected = pd.DataFrame(
#         {
#             "week_num": [1, 1, 2, 2, 3, 3, 4, 4],
#             "article_id": [1, 2, 1, 2, 2, 3, 2, 3],
#         }
#     )
#     pd.testing.assert_frame_equal(
#         result.sort_values(["week_num", "article_id"]).reset_index(drop=True),
#         expected.sort_values(["week_num", "article_id"]).reset_index(drop=True),
#     )


# def test_calculate_rolling_popular_items_with_quantity(sample_transaction_data):
#     """Test calculation using quantity column instead of counts."""
#     result = calculate_rolling_popular_items(
#         df=sample_transaction_data,
#         window_size=1,
#         top_k=2,
#         value_col="quantity",
#     )

#     # Expected results based on quantity:
#     # Week 1: Same as counts
#     # Week 2: article 2 (3 quantity), article 3 (1 quantity)
#     # Week 3: article 1 (1 quantity), article 2 (1 quantity)
#     # Week 4: article 3 (2 quantity)
#     expected = pd.DataFrame(
#         {
#             "week_num": [1, 1, 2, 2, 3, 3, 4, 4],
#             "article_id": [1, 2, 2, 3, 1, 2, 3, 2],
#         }
#     )
#     pd.testing.assert_frame_equal(
#         result.sort_values(["week_num", "article_id"]).reset_index(drop=True),
#         expected.sort_values(["week_num", "article_id"]).reset_index(drop=True),
#     )


# def test_calculate_rolling_popular_items_missing_weeks():
#     """Test handling of missing weeks in the data."""
#     df = pd.DataFrame(
#         {
#             "week_num": [1, 1, 3, 3, 5],  # Week 2 and 4 missing
#             "article_id": [1, 2, 3, 1, 2],
#         }
#     )

#     result = calculate_rolling_popular_items(df, window_size=2, top_k=2)

#     # Should have results for all weeks 1-5
#     assert set(result["week_num"].unique()) == {1, 2, 3, 4, 5}


# def test_calculate_rolling_popular_items_edge_cases():
#     """Test edge cases like empty data or small windows."""
#     # Empty DataFrame
#     empty_df = pd.DataFrame(columns=["week_num", "article_id"])
#     with pytest.raises(ValueError):
#         calculate_rolling_popular_items(empty_df, window_size=1, top_k=1)

#     # Single row
#     single_row_df = pd.DataFrame({"week_num": [1], "article_id": [1]})
#     result = calculate_rolling_popular_items(single_row_df, window_size=1, top_k=1)
#     assert len(result) == 1
#     assert result.iloc[0]["article_id"] == 1

#     # Window size larger than data range
#     result = calculate_rolling_popular_items(single_row_df, window_size=5, top_k=1)
#     assert len(result) == 1
#     assert result.iloc[0]["article_id"] == 1


# def test_calculate_weekly_popular_items(weekly_transaction_data):
#     """Test weekly popular items calculation."""
#     result = calculate_weekly_popular_items(
#         df=weekly_transaction_data,
#         top_k=2,  # Get top 2 items per week
#     )

#     # Expected results:
#     # Week 1: article 1 (2 sales, rank 1), article 2 (1 sale, rank 2)
#     # Week 2: article 2 (2 sales, rank 1), article 1/3 (1 sale each, rank 2)
#     expected = pd.DataFrame(
#         {
#             "week_num": [1, 1, 2, 2, 2],
#             "article_id": [1, 2, 2, 1, 3],
#             "bestseller_rank": [1.0, 2.0, 1.0, 2.0, 2.0],
#         }
#     )

#     # Sort both dataframes to ensure consistent comparison
#     result = result.sort_values(["week_num", "article_id"]).reset_index(drop=True)
#     expected = expected.sort_values(["week_num", "article_id"]).reset_index(drop=True)

#     pd.testing.assert_frame_equal(result, expected)


# def test_calculate_weekly_popular_items_custom_columns(weekly_transaction_data):
#     """Test weekly popular items calculation with custom column names."""
#     # Rename columns in test data
#     df = weekly_transaction_data.rename(columns={"week_num": "week", "article_id": "item_id"})

#     result = calculate_weekly_popular_items(
#         df=df,
#         top_k=2,
#         week_col="week",
#         article_col="item_id",
#     )

#     assert "week" in result.columns
#     assert "item_id" in result.columns
#     assert "bestseller_rank" in result.columns


# def test_calculate_article_weekly_stats(weekly_transaction_data):
#     """Test article weekly statistics calculation."""
#     result = calculate_article_weekly_stats(df=weekly_transaction_data)

#     # Verify the structure and content of the results
#     assert set(result.columns) == {"week_num", "article_id", "count", "avg_price", "mode_sales_channel_id"}

#     # Check specific values for week 1, article 1
#     week1_article1 = result[(result["week_num"] == 1) & (result["article_id"] == 1)].iloc[0]
#     assert week1_article1["count"] == 2  # 2 sales
#     assert week1_article1["avg_price"] == 10.0  # Average of [10.0, 10.0]
#     assert week1_article1["mode_sales_channel_id"] == 1  # Most common channel

#     # Check specific values for week 2, article 2
#     week2_article2 = result[(result["week_num"] == 2) & (result["article_id"] == 2)].iloc[0]
#     assert week2_article2["count"] == 2  # 2 sales
#     assert week2_article2["avg_price"] == 25.0  # Average of [25.0, 25.0]
#     assert week2_article2["mode_sales_channel_id"] == 2  # Most common channel


# def test_calculate_article_weekly_stats_custom_columns(weekly_transaction_data):
#     """Test article weekly stats calculation with custom column names."""
#     # Rename columns in test data
#     df = weekly_transaction_data.rename(
#         columns={
#             "week_num": "week",
#             "article_id": "item_id",
#             "t_dat": "date",
#             "price": "unit_price",
#             "sales_channel_id": "channel",
#         }
#     )

#     result = calculate_article_weekly_stats(
#         df=df, week_col="week", article_col="item_id", date_col="date", price_col="unit_price", channel_col="channel"
#     )

#     assert set(result.columns) == {"week", "item_id", "count", "avg_price", "mode_sales_channel_id"}


# def test_calculate_article_weekly_stats_empty_data():
#     """Test article weekly stats calculation with empty data."""
#     empty_df = pd.DataFrame(columns=["week_num", "article_id", "t_dat", "price", "sales_channel_id"])
#     result = calculate_article_weekly_stats(df=empty_df)
#     assert len(result) == 0
#     assert set(result.columns) == {"week_num", "article_id", "count", "avg_price", "mode_sales_channel_id"}


def test_prepare_negative_samples_by_previous_purchases_basic(purchase_history_data, expected_negative_samples):
    """Test basic functionality and accuracy of NegativeSampleGenerator class."""
    negative_sample_generator = NegativeSampleGenerator()
    result = negative_sample_generator.generate(
        transactions=purchase_history_data,
        week_num_start=0,
        week_num_end=100,
        default_future_week=100,
    )

    # Verify structure
    expected_columns = {"customer_id", "week_num", "article_id", "price", "sales_channel_id", "source"}
    assert set(result.columns) == expected_columns
    assert result["source"].unique() == ["repurchase"]

    # Verify week mapping
    # Customer 1: week 0 -> 1, week 1 -> 2, week 2 -> 4, week 4 -> 100 (validation)
    # Customer 2: week 1 -> 3, week 3 -> 100 (validation)
    # Customer 3: week 2 -> 100 (validation)
    customer1_weeks = result[result["customer_id"] == 1]["week_num"].unique()
    assert set(customer1_weeks) & {1, 2, 4, 100} == set(customer1_weeks)

    # Verify correctness of dataframe
    assert_frame_equal(result, expected_negative_samples)


@pytest.mark.parametrize("customers_ids", ([1], [1, 2]))
def test_prepare_negative_samples_by_previous_purchases_customer_filter(
    purchase_history_data, expected_negative_samples, customers_ids
):
    """Test filtering by customer IDs of NegativeSampleGenerator class."""
    negative_sample_generator = NegativeSampleGenerator()
    result = negative_sample_generator.generate(
        transactions=purchase_history_data,
        week_num_start=0,
        week_num_end=100,
        default_future_week=100,
        customers_ids=customers_ids,
    )
    expected_negative_samples_customer_filter = expected_negative_samples[
        expected_negative_samples["customer_id"].isin(customers_ids)
    ]

    assert set(result["customer_id"].unique()) == set(customers_ids)
    assert_frame_equal(result, expected_negative_samples_customer_filter)


@pytest.mark.parametrize(
    "week_num_start, week_num_end",
    [
        (0, 100),
        (1, 100),
        (1, 5),
    ],
)
def test_prepare_negative_samples_by_previous_purchases_week_filter(
    purchase_history_data, expected_negative_samples, week_num_start, week_num_end
):
    """Test filtering by week range of NegativeSampleGenerator class."""
    negative_sample_generator = NegativeSampleGenerator()
    result = negative_sample_generator.generate(
        transactions=purchase_history_data,
        week_num_start=week_num_start,
        week_num_end=week_num_end,
        default_future_week=100,
    )
    expected_negative_samples_range = expected_negative_samples[
        expected_negative_samples["week_num"].between(week_num_start, week_num_end)
    ]

    assert_frame_equal(result, expected_negative_samples_range)


@pytest.mark.parametrize(
    "input_data, required_cols, should_raise",
    [
        # Valid case - all required columns present
        (
            pd.DataFrame(
                {
                    "customer_id": [1],
                    "week_num": [1],
                    "article_id": [1],
                    "price": [10.0],
                    "sales_channel_id": [1],
                }
            ),
            ["customer_id", "week_num", "article_id"],
            False,
        ),
        # Missing columns case
        (
            pd.DataFrame({"customer_id": [1], "week_num": [1]}),
            ["customer_id", "week_num", "article_id"],
            True,
        ),
        # Empty DataFrame with correct columns
        (
            pd.DataFrame(columns=["customer_id", "week_num", "article_id"]),
            ["customer_id", "week_num", "article_id"],
            False,
        ),
    ],
)
def test_validate_input(input_data, required_cols, should_raise):
    """Test input validation with different scenarios."""
    if should_raise:
        with pytest.raises(ValueError):
            NegativeSampleGenerator._validate_input(input_data, required_cols)
    else:
        NegativeSampleGenerator._validate_input(input_data, required_cols)


@pytest.mark.parametrize(
    "input_data, customer_ids, expected_rows",
    [
        # Filter single customer
        (
            pd.DataFrame(
                {
                    "customer_id": [1, 2, 3],
                    "value": ["a", "b", "c"],
                }
            ),
            [1],
            1,
        ),
        # Filter multiple customers
        (
            pd.DataFrame(
                {
                    "customer_id": [1, 2, 3, 4],
                    "value": ["a", "b", "c", "d"],
                }
            ),
            [1, 3],
            2,
        ),
        # No filter (None)
        (
            pd.DataFrame(
                {
                    "customer_id": [1, 2, 3],
                    "value": ["a", "b", "c"],
                }
            ),
            None,
            3,
        ),
        # Empty customer list
        (
            pd.DataFrame(
                {
                    "customer_id": [1, 2, 3],
                    "value": ["a", "b", "c"],
                }
            ),
            [],
            0,
        ),
    ],
)
def test_filter_customers(input_data, customer_ids, expected_rows):
    """Test customer filtering with different scenarios."""
    result = NegativeSampleGenerator._filter_customers(input_data, customer_ids)
    assert len(result) == expected_rows
    if customer_ids is not None and len(customer_ids) > 0:
        assert set(result["customer_id"].unique()) == set(customer_ids)


@pytest.mark.parametrize(
    "input_data, default_future_week, expected_mappings",
    [
        # Single customer, sequential weeks
        (
            pd.DataFrame(
                {
                    "customer_id": [1, 1, 1],
                    "week_num": [1, 2, 3],
                }
            ),
            100,
            {(1, 1): 2, (1, 2): 3, (1, 3): 100},
        ),
        # Multiple customers
        (
            pd.DataFrame(
                {
                    "customer_id": [1, 1, 2, 2],
                    "week_num": [1, 2, 1, 3],
                }
            ),
            100,
            {(1, 1): 2, (1, 2): 100, (2, 1): 3, (2, 3): 100},
        ),
        # Single week per customer
        (
            pd.DataFrame(
                {
                    "customer_id": [1, 2],
                    "week_num": [1, 1],
                }
            ),
            100,
            {(1, 1): 100, (2, 1): 100},
        ),
    ],
)
def test_create_week_mapping(input_data, default_future_week, expected_mappings):
    """Test week mapping creation with different scenarios."""
    result = NegativeSampleGenerator._create_week_mapping(input_data, default_future_week)
    assert result == expected_mappings


@pytest.mark.parametrize(
    "input_data, required_cols, week_mapping, week_start, week_end, expected_shape, expected_weeks",
    [
        # Basic case
        (
            pd.DataFrame(
                {
                    "customer_id": [1, 1, 2],
                    "week_num": [1, 2, 1],
                    "article_id": [100, 101, 200],
                    "price": [10.0, 20.0, 15.0],
                    "sales_channel_id": [1, 2, 1],
                }
            ),
            ["customer_id", "week_num", "article_id", "price", "sales_channel_id"],
            {(1, 1): 2, (1, 2): 4, (2, 1): 4},
            1,
            4,
            (3, 5),
            {2, 4, 4},
        ),
        # Filter by week range
        (
            pd.DataFrame(
                {
                    "customer_id": [1, 1, 2],
                    "week_num": [1, 2, 1],
                    "article_id": [100, 101, 200],
                    "price": [10.0, 20.0, 15.0],
                    "sales_channel_id": [1, 2, 1],
                }
            ),
            ["customer_id", "week_num", "article_id", "price", "sales_channel_id"],
            {(1, 1): 2, (1, 2): 4, (2, 1): 4},
            2,
            3,
            (1, 5),
            {2},
        ),
    ],
)
def test_apply_week_mapping(
    input_data, required_cols, week_mapping, week_start, week_end, expected_shape, expected_weeks
):
    """Test week mapping application with different scenarios."""
    result = NegativeSampleGenerator._apply_week_mapping(input_data, required_cols, week_mapping, week_start, week_end)

    # Check shape and columns
    assert result.shape == expected_shape
    assert set(result.columns) == set(required_cols)

    # Check week range
    assert set(result["week_num"].unique()) == expected_weeks

    # Check data types
    assert result["week_num"].dtype == np.int64
