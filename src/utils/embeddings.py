"""Utility functions for handling embeddings."""

import logging
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

from src.features_articles import ArticleEmbeddingResult

logger = logging.getLogger(__name__)


def get_memory_usage() -> Tuple[float, float]:
    """Get current memory usage of the process.

    Returns:
        Tuple of (memory_usage_gb, memory_percent)
    """
    process = psutil.Process()
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024  # Convert bytes to GB
    memory_percent = process.memory_percent()
    return memory_gb, memory_percent


def calculate_cosine_similarity_batch(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between two sets of embeddings using vectorized operations.

    Args:
        embeddings1: Array of shape (N, embedding_dim) containing first set of embeddings
        embeddings2: Array of shape (N, embedding_dim) containing second set of embeddings

    Returns:
        Array of shape (N,) containing cosine similarities
    """
    # Use float32 instead of float64 to reduce memory usage
    embeddings1 = embeddings1.astype(np.float32)
    embeddings2 = embeddings2.astype(np.float32)

    # Normalize the embeddings (along axis 1 which is the embedding dimension)
    embeddings1_norm = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2_norm = np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # Avoid division by zero
    embeddings1_norm = np.where(embeddings1_norm == 0, 1e-8, embeddings1_norm)
    embeddings2_norm = np.where(embeddings2_norm == 0, 1e-8, embeddings2_norm)

    # Normalize in-place to save memory
    np.divide(embeddings1, embeddings1_norm, out=embeddings1)
    np.divide(embeddings2, embeddings2_norm, out=embeddings2)

    # Calculate cosine similarity using dot product of normalized embeddings
    # Use einsum for more efficient matrix multiplication
    return np.einsum("ij,ij->i", embeddings1, embeddings2)


def calculate_df_batch_cosine_similarity(
    df: pd.DataFrame,
    article_embeddings: ArticleEmbeddingResult,
    article_embedding_type: str = "text",
    customer_text_embedding_col: str = "customer_avg_text_embedding",
    batch_size: int = 10000,
    monitor_memory: bool = True,
) -> np.ndarray:
    """Calculate cosine similarity between customer and article embeddings in batches.

    Args:
        df: DataFrame containing article_id and customer text embedding column
        article_embeddings: ArticleEmbeddingResult containing article embeddings
        customer_text_embedding_col: Name of the column containing customer text embeddings
        batch_size: Size of batches to process at once
        monitor_memory: Whether to monitor and log memory usage

    Returns:
        Array of cosine similarities
    """
    if article_embedding_type not in ["text", "image"]:
        raise ValueError(f"Invalid article embedding type: {article_embedding_type}")

    start_time = time.time()
    if monitor_memory:
        mem_gb, mem_percent = get_memory_usage()
        logger.info(f"Initial memory usage: {mem_gb:.2f} GB ({mem_percent:.1f}%)")

    # Get article embeddings for all articles in the DataFrame
    article_indices = np.array(
        [
            article_embeddings.id_to_index[article_id]
            for article_id in df["article_id"]
            if article_id in article_embeddings.id_to_index
        ]
    )

    if len(article_indices) == 0:
        return np.full(len(df), np.nan)

    # Get embedding dimension for zero vectors
    embedding_dim = article_embeddings.text_embeddings.shape[1]

    if monitor_memory:
        mem_gb, mem_percent = get_memory_usage()
        logger.info(f"Memory usage after loading article indices: {mem_gb:.2f} GB ({mem_percent:.1f}%)")

    # Process in batches to manage memory
    similarities = np.zeros(len(df), dtype=np.float32)  # Use float32 for output array

    n_batches = (len(df) + batch_size - 1) // batch_size
    for batch_idx, i in tqdm(enumerate(range(0, len(df), batch_size))):
        batch_start_time = time.time()
        batch_end = min(i + batch_size, len(df))

        # Get article embeddings only for this batch
        if article_embedding_type == "text":
            batch_article_embeddings = article_embeddings.text_embeddings[article_indices[i:batch_end]]
        elif article_embedding_type == "image":
            batch_article_embeddings = article_embeddings.image_embeddings[article_indices[i:batch_end]]

        # Convert customer embeddings to numpy array only for this batch
        batch_customer_embeddings = np.array(
            [
                emb if isinstance(emb, np.ndarray) else np.zeros(embedding_dim)
                for emb in df[customer_text_embedding_col].iloc[i:batch_end]
            ],
            dtype=np.float32,
        )

        similarities[i:batch_end] = calculate_cosine_similarity_batch(
            batch_customer_embeddings, batch_article_embeddings
        )

        if monitor_memory:
            batch_time = time.time() - batch_start_time
            mem_gb, mem_percent = get_memory_usage()
            logger.info(
                f"Batch {batch_idx + 1}/{n_batches} processed in {batch_time:.2f}s. "
                f"Memory usage: {mem_gb:.2f} GB ({mem_percent:.1f}%)"
            )

    total_time = time.time() - start_time
    if monitor_memory:
        mem_gb, mem_percent = get_memory_usage()
        logger.info(
            f"Total processing time: {total_time:.2f}s. " f"Final memory usage: {mem_gb:.2f} GB ({mem_percent:.1f}%)"
        )

    return similarities
