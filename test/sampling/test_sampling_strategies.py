import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.utils.sampling_strategies import (
    NegativeSampleGenerator,
    PopularityBasedSampler,
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


@pytest.fixture
def sample_transactions_for_manager():
    """Fixture providing sample transaction data for NegativeSamplingManager tests."""
    return pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 3, 3],
            "week_num": [10, 11, 10, 11, 10, 11],
            "article_id": [101, 102, 201, 202, 301, 302],
            "price": [10.0, 20.0, 15.0, 25.0, 30.0, 35.0],
            "sales_channel_id": [1, 2, 1, 2, 1, 2],
            "t_dat": pd.date_range("2020-01-01", periods=6),
        }
    )


@pytest.fixture
def sample_customer_week_pairs():
    """Fixture providing sample customer-week pairs for NegativeSamplingManager tests."""
    return pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 3, 3],
            "week_num": [12, 13, 12, 13, 12, 13],
        }
    )


@pytest.fixture
def mock_sampling_strategies():
    """Fixture providing mock sampling strategies configuration."""
    return {
        "popularity": {"top_k_items": 2},
        "repurchase": {},
    }


def test_negative_sampling_manager_init(mock_sampling_strategies):
    """Test initialization of NegativeSamplingManager."""
    manager = NegativeSamplingManager(sampling_strategies=mock_sampling_strategies)

    # Check attributes
    assert manager.sampling_strategies == mock_sampling_strategies
    assert set(manager.samplers.keys()) == {"popularity", "repurchase"}
    assert isinstance(manager.samplers["popularity"], PopularityBasedSampler)
    assert isinstance(manager.samplers["repurchase"], NegativeSampleGenerator)


def test_negative_sampling_manager_calculate_popular_items(sample_transactions_for_manager, mock_sampling_strategies):
    """Test calculation of popular items."""
    manager = NegativeSamplingManager(sampling_strategies=mock_sampling_strategies)

    popular_items, weekly_stats = manager._calculate_popular_items(
        transactions=sample_transactions_for_manager, week_num_start=10, week_num_end=12, top_k=2
    )

    # Check popular items
    assert set(popular_items.columns) == {"week_num", "article_id", "bestseller_rank"}
    assert len(popular_items) > 0

    # Check weekly stats
    assert set(weekly_stats.columns) == {"week_num", "article_id", "count", "avg_price", "mode_sales_channel_id"}
    assert len(weekly_stats) > 0


def test_negative_sampling_manager_get_default_prediction(sample_transactions_for_manager, mock_sampling_strategies):
    """Test getting default prediction."""
    manager = NegativeSamplingManager(sampling_strategies=mock_sampling_strategies)

    # Calculate popular items first
    popular_items, _ = manager._calculate_popular_items(
        transactions=sample_transactions_for_manager, week_num_start=10, week_num_end=12, top_k=2
    )

    # Get default prediction
    default_prediction = manager._get_default_prediction(popular_items, week_num=10)

    # Check default prediction
    assert default_prediction is not None
    assert isinstance(default_prediction, np.ndarray)
    assert len(default_prediction) > 0


def test_negative_sampling_manager_generate_popularity_samples(
    sample_transactions_for_manager, sample_customer_week_pairs, mock_sampling_strategies
):
    """Test generation of popularity-based samples."""
    manager = NegativeSamplingManager(sampling_strategies=mock_sampling_strategies)

    # Calculate popular items first
    popular_items, weekly_stats = manager._calculate_popular_items(
        transactions=sample_transactions_for_manager, week_num_start=10, week_num_end=12, top_k=2
    )

    # Generate popularity samples
    samples = manager._generate_popularity_samples(
        unique_customer_week_pairs=sample_customer_week_pairs, popular_items=popular_items, weekly_stats=weekly_stats
    )

    # Check samples
    assert len(samples) > 0
    assert set(samples.columns) >= {"customer_id", "week_num", "article_id", "source"}
    assert samples["source"].unique() == ["popularity"]


def test_negative_sampling_manager_generate_repurchase_samples(
    sample_transactions_for_manager, mock_sampling_strategies
):
    """Test generation of repurchase-based samples."""
    manager = NegativeSamplingManager(sampling_strategies=mock_sampling_strategies)

    # Generate repurchase samples
    samples = manager._generate_repurchase_samples(
        transactions=sample_transactions_for_manager, week_num_start=11, week_num_end=13
    )

    # Check samples
    assert len(samples) > 0
    assert set(samples.columns) >= {"customer_id", "week_num", "article_id", "source"}
    assert samples["source"].unique() == ["repurchase"]


@pytest.mark.parametrize("sample_type", ["train", "inference"])
def test_negative_sampling_manager_combine_samples(
    sample_transactions_for_manager, mock_sampling_strategies, sample_type
):
    """Test combining samples from different sources."""
    manager = NegativeSamplingManager(sampling_strategies=mock_sampling_strategies)

    # Create mock samples
    popularity_samples = pd.DataFrame(
        {
            "customer_id": [1, 1, 2],
            "week_num": [12, 12, 12],
            "article_id": [101, 102, 201],
            "price": [10.0, 20.0, 15.0],
            "sales_channel_id": [1, 2, 1],
            "source": ["popularity"] * 3,
        }
    )

    repurchase_samples = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "week_num": [12, 12, 12],
            "article_id": [101, 201, 301],  # Note: Some overlap with popularity
            "price": [10.0, 15.0, 30.0],
            "sales_channel_id": [1, 1, 1],
            "source": ["repurchase"] * 3,
        }
    )

    # Combine samples
    combined = manager.combine_samples(
        transactions=sample_transactions_for_manager,
        list_candidates=[popularity_samples, repurchase_samples],
        sample_type=sample_type,
    )

    # Check combined samples
    assert len(combined) > 0
    # Should be fewer than the sum due to deduplication
    assert len(combined) < len(popularity_samples) + len(repurchase_samples)


def test_negative_sampling_manager_generate_samples(
    sample_transactions_for_manager, sample_customer_week_pairs, mock_sampling_strategies
):
    """Test end-to-end sample generation."""
    manager = NegativeSamplingManager(sampling_strategies=mock_sampling_strategies)

    # Generate samples
    samples, default_prediction, popular_items = manager.generate_samples(
        transactions=sample_transactions_for_manager,
        unique_customer_week_pairs=sample_customer_week_pairs,
        week_num_start=10,
        week_num_end=13,
        sample_type="train",
    )

    # Check samples
    assert len(samples) > 0
    assert set(samples.columns) >= {"customer_id", "week_num", "article_id", "source"}

    # Check sources
    assert set(samples["source"].unique()) >= {"popularity", "repurchase"}

    # Check popular items
    assert len(popular_items) > 0

    # Check default_prediction is None for train
    assert default_prediction is None
