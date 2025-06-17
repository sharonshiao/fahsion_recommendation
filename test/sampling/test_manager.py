import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.sampling.manager import NegativeSamplingManager
from src.sampling.popularity import PopularityBasedSampler
from src.sampling.repurchase import RepurchaseSampleGenerator


@pytest.fixture
def sample_transactions_for_manager():
    """Fixture providing sample transaction data for NegativeSamplingManager tests."""
    return pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 3, 3],
            "week_num": [11, 12, 11, 12, 11, 12],
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


# FIX: we currently only have test cases for last_purchase strategy. To add test cases for last_k_items strategy,
@pytest.fixture
def mock_sampling_strategies():
    """Fixture providing mock sampling strategies configuration."""
    return {
        "popularity": {"top_k_items": 2},
        "repurchase": {"strategy": "last_purchase", "k": 12},
    }


def test_negative_sampling_manager_init(mock_sampling_strategies):
    """Test initialization of NegativeSamplingManager."""
    manager = NegativeSamplingManager(sampling_strategies=mock_sampling_strategies)

    # Check attributes
    assert manager.sampling_strategies == mock_sampling_strategies
    assert set(manager.samplers.keys()) == {"popularity", "repurchase"}
    assert isinstance(manager.samplers["popularity"], PopularityBasedSampler)
    assert isinstance(manager.samplers["repurchase"], RepurchaseSampleGenerator)


def test_negative_sampling_manager_calculate_popular_items(sample_transactions_for_manager, mock_sampling_strategies):
    """Test calculation of popular items."""
    manager = NegativeSamplingManager(sampling_strategies=mock_sampling_strategies)

    popular_items, weekly_stats = manager._calculate_popular_items(
        transactions=sample_transactions_for_manager, week_num_start=12, week_num_end=13, top_k=2
    )

    # Check popular items
    assert set(popular_items.columns) == {"week_num", "article_id", "bestseller_rank"}
    assert len(popular_items) == 4

    # Check weekly stats
    assert set(weekly_stats.columns) == {"week_num", "article_id", "count", "avg_price", "mode_sales_channel_id"}
    assert len(weekly_stats) == 6


def test_negative_sampling_manager_get_default_prediction(sample_transactions_for_manager, mock_sampling_strategies):
    """Test getting default prediction."""
    manager = NegativeSamplingManager(sampling_strategies=mock_sampling_strategies)

    # Calculate popular items first
    popular_items, _ = manager._calculate_popular_items(
        transactions=sample_transactions_for_manager, week_num_start=12, week_num_end=13, top_k=2
    )

    # Get default prediction
    default_prediction = manager._get_default_prediction(popular_items, week_num=12)
    expected_default_prediction = np.array([102, 202])

    # Check default prediction
    assert default_prediction is not None
    assert isinstance(default_prediction, np.ndarray)
    assert len(default_prediction) > 0
    assert np.array_equal(default_prediction, expected_default_prediction)


def test_negative_sampling_manager_generate_popularity_samples(
    sample_transactions_for_manager, sample_customer_week_pairs, mock_sampling_strategies
):
    """Test generation of popularity-based samples."""
    manager = NegativeSamplingManager(sampling_strategies=mock_sampling_strategies)

    # Calculate popular items first
    popular_items, weekly_stats = manager._calculate_popular_items(
        transactions=sample_transactions_for_manager, week_num_start=12, week_num_end=13, top_k=2
    )

    # Generate popularity samples
    samples = manager._generate_popularity_samples(
        unique_customer_week_pairs=sample_customer_week_pairs, popular_items=popular_items, weekly_stats=weekly_stats
    )

    # Check samples
    assert len(samples) == 12
    assert set(samples.columns) == {
        "customer_id",
        "week_num",
        "article_id",
        "source",
        "sales_channel_id",
        "price",
    }
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
    assert set(samples.columns) == {"customer_id", "week_num", "article_id", "source", "price", "sales_channel_id"}
    assert samples["source"].unique() == ["repurchase"]


@pytest.mark.parametrize(
    "popularity_samples, repurchase_samples, expected_combined_samples",
    [
        # Note: the negative samples does not refer to any real transactions record and only used for testing
        # the combination logic.
        (
            pd.DataFrame(
                {
                    "customer_id": [1, 1, 2],
                    "week_num": [12, 12, 12],
                    "article_id": [101, 102, 201],
                    "price": [10.0, 20.0, 15.0],
                    "sales_channel_id": [1, 2, 1],
                    "source": ["popularity"] * 3,
                }
            ),
            pd.DataFrame(
                {
                    "customer_id": [1, 2, 3],
                    "week_num": [12, 12, 12],
                    "article_id": [101, 201, 302],
                    "price": [10.0, 15.0, 30.0],
                    "sales_channel_id": [1, 1, 1],
                    "source": ["repurchase"] * 3,
                }
            ),
            pd.DataFrame(
                {
                    "customer_id": [1, 2],
                    "week_num": [12, 12],
                    "article_id": [101, 201],
                    "price": [10.0, 15.0],
                    "sales_channel_id": [1, 1],
                    "source": ["popularity"] * 2,
                }
            ),
        )
    ],
)
def test_negative_sampling_manager_combine_samples(
    sample_transactions_for_manager,
    mock_sampling_strategies,
    popularity_samples,
    repurchase_samples,
    expected_combined_samples,
):
    """Test combining samples from different sources."""
    manager = NegativeSamplingManager(sampling_strategies=mock_sampling_strategies)

    # Combine samples
    combined = manager.combine_samples(
        transactions=sample_transactions_for_manager,
        list_candidates=[popularity_samples, repurchase_samples],
        sample_type="train",
    )

    # Check combined samples
    assert_frame_equal(combined, expected_combined_samples)


@pytest.mark.parametrize(
    "sample_type, expected_default_prediction",
    [("train", None), ("inference", np.array([101, 201]))],
)
def test_negative_sampling_manager_generate_samples(
    sample_transactions_for_manager,
    sample_customer_week_pairs,
    mock_sampling_strategies,
    sample_type,
    expected_default_prediction,
):
    """Test end-to-end sample generation."""
    manager = NegativeSamplingManager(sampling_strategies=mock_sampling_strategies)

    # Generate samples
    samples, default_prediction, popular_items = manager.generate_samples(
        transactions=sample_transactions_for_manager,
        unique_customer_week_pairs=sample_customer_week_pairs,
        week_num_start=12,
        week_num_end=13,
        sample_type=sample_type,
    )

    # Check samples
    assert len(samples) > 0
    assert set(samples.columns) >= {"customer_id", "week_num", "article_id", "source"}

    # Check sources
    assert set(samples["source"].unique()) >= {"popularity", "repurchase"}

    # Check popular items
    assert len(popular_items) > 0

    # Check default_prediction is None for train
    if sample_type == "train":
        assert default_prediction is None
    else:
        assert default_prediction is not None
        assert np.array_equal(default_prediction, expected_default_prediction)
