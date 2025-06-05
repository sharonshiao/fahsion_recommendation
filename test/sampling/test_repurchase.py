import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.sampling.repurchase import RepurchaseSampleGenerator


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


def test_prepare_negative_samples_by_previous_purchases_basic(purchase_history_data, expected_negative_samples):
    """Test basic functionality and accuracy of NegativeSampleGenerator class."""
    negative_sample_generator = RepurchaseSampleGenerator()
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
    negative_sample_generator = RepurchaseSampleGenerator()
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
    negative_sample_generator = RepurchaseSampleGenerator()
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
            RepurchaseSampleGenerator._validate_input(input_data, required_cols)
    else:
        RepurchaseSampleGenerator._validate_input(input_data, required_cols)


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
    result = RepurchaseSampleGenerator._filter_customers(input_data, customer_ids)
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
    result = RepurchaseSampleGenerator._create_week_mapping(input_data, default_future_week)
    expected_mappings = pd.DataFrame.from_dict(expected_mappings, columns=["next_week"], orient="index").reset_index()
    expected_mappings["customer_id"], expected_mappings["week_num"] = zip(*expected_mappings["index"])
    expected_mappings.drop(columns=["index"], inplace=True)
    # Rearrange columns
    expected_mappings = expected_mappings[["customer_id", "week_num", "next_week"]]
    assert_frame_equal(result, expected_mappings)


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
            pd.DataFrame(
                {
                    "customer_id": [1, 1, 2],
                    "week_num": [1, 2, 1],
                    "next_week": [2, 4, 4],
                }
            ),
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
            pd.DataFrame(
                {
                    "customer_id": [1, 1, 2],
                    "week_num": [1, 2, 1],
                    "next_week": [2, 4, 4],
                }
            ),
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
    result = RepurchaseSampleGenerator._apply_week_mapping(
        input_data, required_cols, week_mapping, week_start, week_end
    )

    # Check shape and columns
    assert result.shape == expected_shape
    assert set(result.columns) == set(required_cols)

    # Check week range
    assert set(result["week_num"].unique()) == expected_weeks

    # Check data types
    assert result["week_num"].dtype == np.int64
