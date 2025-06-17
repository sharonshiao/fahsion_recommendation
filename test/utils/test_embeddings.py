import numpy as np
import pandas as pd
import pytest

from src.features_articles import ArticleEmbeddingResult
from src.utils.embeddings import (
    calculate_cosine_similarity_batch,
    calculate_df_batch_cosine_similarity,
)


@pytest.mark.parametrize(
    "embeddings1, embeddings2, expected_result",
    [
        (np.array([[1, 2, 3], [1, 1, 1]]), np.array([[1, 1, 1], [2, -2, -2]]), np.array([0.9258201, -1 / 3])),
        (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [4, 5, 6]]), np.array([1.0, 1.0])),
        (np.array([[1, 0], [0, 1]]), np.array([[1, 0], [1, 0]]), np.array([1.0, 0.0])),
    ],
)
def test_calculate_cosine_similarity_batch(
    embeddings1: np.ndarray, embeddings2: np.ndarray, expected_result: np.ndarray
):
    result = calculate_cosine_similarity_batch(embeddings1, embeddings2)
    assert np.allclose(result, expected_result)


@pytest.fixture
def sample_embeddings_df():
    """Fixture providing a sample DataFrame with customer embeddings."""
    return pd.DataFrame(
        {
            "article_id": [1, 2, 3, 4],
            "customer_avg_text_embedding": [
                np.array([1.0, 0.0]),  # Normal embedding
                np.array([0.0, 1.0]),  # Normal embedding
                None,  # Missing embedding
                np.array([1.0, 1.0]),  # Normal embedding
            ],
        }
    )


@pytest.fixture
def mock_article_embeddings():
    """Fixture providing mock article embeddings."""
    text_embeddings = np.array(
        [
            [1.0, 0.0],  # Article 1 embedding
            [0.0, 1.0],  # Article 2 embedding
            [1.0, 1.0],  # Article 3 embedding
            [-1.0, 1.0],  # Article 4 embedding
        ],
        dtype=np.float32,
    )

    return ArticleEmbeddingResult(
        text_embeddings=text_embeddings,
        image_embeddings=text_embeddings.copy(),  # Use same embeddings for image
        image_missing=np.zeros(4, dtype=bool),
        id_to_index={1: 0, 2: 1, 3: 2, 4: 3},
        index_to_id={0: 1, 1: 2, 2: 3, 3: 4},
    )


@pytest.fixture
def expected_similarities():
    """Fixture providing expected similarity values."""
    return np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)


@pytest.mark.parametrize(
    "embedding_type,expected_error",
    [
        ("text", None),
        ("image", None),
        ("invalid", ValueError),
    ],
)
def test_calculate_df_batch_cosine_similarity(
    sample_embeddings_df,
    mock_article_embeddings,
    expected_similarities,
    embedding_type,
    expected_error,
):
    """Test batch cosine similarity calculation with different embedding types.

    Test cases:
    1. Normal embeddings matching exactly (cosine similarity = 1.0)
    2. Missing embeddings replaced with zeros (cosine similarity = 0.0)
    3. Orthogonal embeddings (cosine similarity = 0.0)
    4. Different batch sizes
    5. Error handling for invalid embedding types

    Args:
        sample_embeddings_df: DataFrame with test customer embeddings
        mock_article_embeddings: Mock article embeddings
        expected_similarities: Expected similarity values
        embedding_type: Type of embedding to test
        expected_error: Expected error type, if any
    """
    if expected_error is not None:
        with pytest.raises(expected_error):
            calculate_df_batch_cosine_similarity(
                df=sample_embeddings_df,
                article_embeddings=mock_article_embeddings,
                article_embedding_type=embedding_type,
                customer_embedding_col="customer_avg_text_embedding",
            )
    else:
        # Test with different batch sizes to ensure batching works correctly
        for batch_size in [1, 2, 4]:
            similarities = calculate_df_batch_cosine_similarity(
                df=sample_embeddings_df,
                article_embeddings=mock_article_embeddings,
                article_embedding_type=embedding_type,
                customer_embedding_col="customer_avg_text_embedding",
                batch_size=batch_size,
                monitor_memory=False,
            )

            # Verify results
            assert len(similarities) == len(sample_embeddings_df)
            assert np.allclose(similarities, expected_similarities, atol=1e-6)
