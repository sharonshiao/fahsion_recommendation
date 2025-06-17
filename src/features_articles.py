import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer

from src.config import (
    DEFAULT_ARTICLE_DYNAMIC_FEATURES_CONFIG,
    DEFAULT_ARTICLE_EMBEDDING_CONFIG,
    DEFAULT_ARTICLE_STATIC_FEATURES_CONFIG,
)
from src.feature_extraction import load_optimized_raw_data
from src.utils.utils_torch import get_device

logger = logging.getLogger(__name__)

ARTICLES_FEATURE_TYPES = ["static", "dynamic", "embedding"]


def get_path_to_article_features(feature_type: str, subsample: float, seed: int) -> str:
    """Get path to article features.

    Args:
        feature_type: Type of features (e.g. 'static', 'dynamic')
        subsample: Subsample ratio (0-1)
        seed: Random seed for reproducibility
    """
    if feature_type not in ARTICLES_FEATURE_TYPES:
        raise ValueError(f"Invalid feature type: {feature_type}. Supported types: {ARTICLES_FEATURE_TYPES}")

    if feature_type == "embedding":
        # Only use full data for embedding
        return "../data/preprocessed/articles_embedding/full"

    if subsample < 1:
        return f"../data/preprocessed/articles_{feature_type}/subsample_{subsample}_{seed}"
    else:
        return f"../data/preprocessed/articles_{feature_type}/full"


def concatenate_text_columns(df: pd.DataFrame, cols_text: list[str]) -> list[str]:
    """Concatenate text columns into a single string."""
    return df[cols_text].fillna("").agg(" ".join, axis=1).tolist()


# ======================================================================================================================
# Article processing functions
# ======================================================================================================================
@dataclass
class ArticleStaticFeaturePipelineConfig:
    """Configuration for article feature pipeline processing."""

    # Processor configuration
    config_processor: Dict[str, Any] = field(
        default_factory=lambda: {
            # Feature encoding
            "encoding_strategy": "ordinal",  # Options: None, "ordinal", "one_hot"
            # Feature classification
            "categorical_features": [
                "product_type_no",
                "graphical_appearance_no",
                "colour_group_code",
                "perceived_colour_value_id",
                "perceived_colour_master_id",
                "department_no",
                "index_code",
                "index_group_no",
                "section_no",
                "garment_group_no",
            ],
            "numerical_features": [],
            "one_hot_features": [],
        }
    )

    # I/O Configuration
    subsample: float = 1.0  # No subsampling by default for articles
    seed: int = 42

    # Helper methods
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ArticleStaticFeaturePipelineConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def create_default(cls) -> "ArticleStaticFeaturePipelineConfig":
        """Create default configuration."""
        return cls.from_dict(DEFAULT_ARTICLE_STATIC_FEATURES_CONFIG)


@dataclass
class ArticleStaticFeatureResult:
    """Results from article feature processing."""

    data: pd.DataFrame  # Processed article dataframe
    feature_names: Dict[str, List[str]]  # Features by type
    transformers: Dict[str, any]  # Encoding transformers

    def save(self, path_to_dir: str):
        """Save result to disk."""
        logger.info(f"Saving article feature result to {path_to_dir}")
        # Create directory if it doesn't exist already
        os.makedirs(path_to_dir, exist_ok=True)

        # Save components separately
        self.data.to_parquet(f"{path_to_dir}/data.parquet")
        logger.info(f"Saved data to {path_to_dir}/data.parquet")

        with open(f"{path_to_dir}/feature_names.json", "w") as f:
            json.dump(self.feature_names, f)
        logger.info(f"Saved feature names to {path_to_dir}/feature_names.json")

        with open(f"{path_to_dir}/transformers.pkl", "wb") as f:
            pickle.dump(self.transformers, f)
        logger.info(f"Saved transformers to {path_to_dir}/transformers.pkl")

    @classmethod
    def load(cls, path_to_dir: str):
        """Load result from disk."""
        logger.info(f"Loading article feature result from {path_to_dir}")

        # Load components
        data = pd.read_parquet(f"{path_to_dir}/data.parquet")
        logger.info(f"Loaded data from {path_to_dir}/data.parquet")

        with open(f"{path_to_dir}/feature_names.json", "r") as f:
            feature_names = json.load(f)
        logger.info(f"Loaded feature names from {path_to_dir}/feature_names.json")

        with open(f"{path_to_dir}/transformers.pkl", "rb") as f:
            transformers = pickle.load(f)
        logger.info(f"Loaded transformers from {path_to_dir}/transformers.pkl")

        return cls(data=data, feature_names=feature_names, transformers=transformers)

    def get_feature_list(self, include_id: bool = True) -> List[str]:
        """Get a flattened list of all features."""
        features = []
        for feature_type, feature_list in self.feature_names.items():
            if not include_id and feature_type == "id_columns":
                continue
            features.extend(feature_list)

        return features


class ArticleStaticFeatureProcessor:
    """Process article data and generate features."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transformers = {}
        self._default_categorical_features = [
            "product_type_no",
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
            "perceived_colour_master_id",
            "department_no",
            "index_code",
            "index_group_no",
            "section_no",
            "garment_group_no",
        ]
        self._default_numerical_features = []
        self._default_one_hot_features = []

        self.categorical_features = self.config.get("categorical_features", self._default_categorical_features)
        self.numerical_features = self.config.get("numerical_features", self._default_numerical_features)
        self.one_hot_features = self.config.get("one_hot_features", self._default_one_hot_features)
        self.id_columns = ["article_id"]

        self.encoding_strategy = self.config.get("encoding_strategy", "ordinal")

    def process(self, articles_raw: pd.DataFrame) -> ArticleStaticFeatureResult:
        """
        Main method to process article data end-to-end.

        Args:
            articles_raw: Raw article DataFrame

        Returns:
            ArticleFeatureResult containing processed data and feature metadata
        """
        logger.info("Processing article data")
        articles = articles_raw.copy()

        # Collect feature names
        self._collect_feature_names(articles)

        # Encode categorical variables
        articles = self._encode_categorical_features(articles)

        return ArticleStaticFeatureResult(
            data=articles,
            feature_names=self._get_feature_names_dict(),
            transformers=self.transformers,
        )

    def _encode_categorical_features(self, articles: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features if specified in config."""

        if self.encoding_strategy == "ordinal":
            logger.debug("Applying ordinal encoding")

            for col in self.categorical_features:
                if col in articles.columns:
                    logger.debug(f"Encoding column: {col}")
                    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                    articles[col] = oe.fit_transform(articles[col].astype(str).values.reshape(-1, 1))
                    self.transformers[f"ordinal_{col}"] = oe

        return articles

    def _collect_feature_names(self, articles: pd.DataFrame) -> None:
        """Collect feature names by type."""
        # Default classifications - override with config if needed
        # Only include columns that are actually present
        self.categorical_features = [col for col in self.categorical_features if col in articles.columns]
        self.numerical_features = [col for col in self.numerical_features if col in articles.columns]
        self.one_hot_features = [col for col in self.one_hot_features if col in articles.columns]

        logger.debug(f"Collected {len(self.categorical_features)} categorical features")
        logger.debug(f"Collected {len(self.numerical_features)} numerical features")
        logger.debug(f"Collected {len(self.one_hot_features)} one-hot features")

    def _get_feature_names_dict(self) -> Dict[str, List[str]]:
        """Get feature names by type."""
        return {
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "one_hot_features": self.one_hot_features,
            "id_columns": self.id_columns,
        }


class ArticleStaticFeaturePipeline:
    """End-to-end pipeline for processing article features."""

    def __init__(self, config=None):
        """Initialize pipeline with config."""
        self.config = config or ArticleStaticFeaturePipelineConfig()
        self.processor = None

    def setup(self, config=None):
        """Set up the pipeline with configuration."""
        if config is not None:
            self.config = config

        logger.info("Setting up ArticleFeaturePipeline with config:")
        logger.debug(json.dumps(asdict(self.config), indent=2, default=str))
        self.processor = ArticleStaticFeatureProcessor(self.config.config_processor)
        return self

    def _load_raw_data(self):
        logger.info("Loading raw article data for ArticleFeaturePipeline")

        articles = load_optimized_raw_data("articles", subsample=self.config.subsample, seed=self.config.seed)
        logger.debug(f"Loaded raw article data with shape: {articles.shape}")
        return articles

    def run(self):
        """Run the pipeline to process article data."""
        if self.processor is None:
            raise ValueError("Pipeline not set up. Call setup() before running the pipeline.")

        articles_raw = self._load_raw_data()

        # Process data
        results = self.processor.process(articles_raw)

        # Save results
        path_to_dir = get_path_to_article_features("static", self.config.subsample, self.config.seed)
        results.save(path_to_dir)

        return results


# ======================================================================================================================
# Article dynamic features
# ======================================================================================================================
@dataclass
class ArticleDynamicFeaturePipelineConfig:
    """Configuration for article dynamic feature pipeline processing."""

    # Feature configuration
    config_processor: Dict[str, Any] = field(
        default_factory=lambda: {
            # Feature encoding
            "encoding_strategy": "ordinal",  # Options: None, "ordinal", "one_hot"
            # Feature classification
            "categorical_features": [],
            "numerical_features": [
                "weekly_sales_count",
                "weekly_avg_price",
                "cumulative_mean_age",
                "cumulative_sales_count",
            ],
            "one_hot_features": [],
            # Dates
            "start_week_num": 76,
            "end_week_num": 104,
            "history_start_week_num": 76,
            "history_end_week_num": 104,
        }
    )

    # I/O Configuration
    subsample: float = 1.0
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ArticleDynamicFeaturePipelineConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def create_default(cls) -> "ArticleDynamicFeaturePipelineConfig":
        """Create default configuration."""
        return cls.from_dict(DEFAULT_ARTICLE_DYNAMIC_FEATURES_CONFIG)


@dataclass
class ArticleDynamicFeatureResult:
    """Results from article dynamic feature processing."""

    data: pd.DataFrame  # Processed article dataframe with dynamic features
    feature_names: Dict[str, List[str]]  # Features by type
    transformers: Dict[str, any] = field(default_factory=dict)  # Any transformers used

    def save(self, path_to_dir: str):
        """Save result to disk."""
        logger.info(f"Saving article dynamic feature result to {path_to_dir}")
        os.makedirs(path_to_dir, exist_ok=True)

        # Save components separately
        self.data.to_parquet(f"{path_to_dir}/data.parquet")
        logger.info(f"Saved data to {path_to_dir}/data.parquet")

        with open(f"{path_to_dir}/feature_names.json", "w") as f:
            json.dump(self.feature_names, f)
        logger.info(f"Saved feature names to {path_to_dir}/feature_names.json")

        if self.transformers:
            with open(f"{path_to_dir}/transformers.pkl", "wb") as f:
                pickle.dump(self.transformers, f)
            logger.info(f"Saved transformers to {path_to_dir}/transformers.pkl")

    @classmethod
    def load(cls, path_to_dir: str) -> "ArticleDynamicFeatureResult":
        """Load result from disk."""
        logger.info(f"Loading article dynamic feature result from {path_to_dir}")

        # Load components
        data = pd.read_parquet(f"{path_to_dir}/data.parquet")
        logger.info(f"Loaded data from {path_to_dir}/data.parquet")

        with open(f"{path_to_dir}/feature_names.json", "r") as f:
            feature_names = json.load(f)
        logger.info(f"Loaded feature names from {path_to_dir}/feature_names.json")

        transformers = {}
        if os.path.exists(f"{path_to_dir}/transformers.pkl"):
            with open(f"{path_to_dir}/transformers.pkl", "rb") as f:
                transformers = pickle.load(f)
            logger.info(f"Loaded transformers from {path_to_dir}/transformers.pkl")

        return cls(data=data, feature_names=feature_names, transformers=transformers)

    def get_feature_list(self, include_id: bool = True) -> List[str]:
        """Get a flattened list of all features."""
        features = []
        for feature_type, feature_list in self.feature_names.items():
            if not include_id and feature_type == "id_columns":
                continue
            features.extend(feature_list)
        return features


class ArticleDynamicFeatureProcessor:
    """Process article data to generate dynamic features based on transaction history."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize processor with configuration."""
        self.config = config
        self.start_week_num = self.config.get("start_week_num", None)
        self.end_week_num = self.config.get("end_week_num", None)
        self.history_start_week_num = self.config.get("history_start_week_num", None)
        self.history_end_week_num = self.config.get("history_end_week_num", None)

        # Features
        self._default_categorical_features = []
        self._default_numerical_features = [
            "weekly_sales_count",
            "weekly_avg_price",
            "cumulative_mean_age",
            "cumulative_sales_count",
        ]
        self._default_one_hot_features = []
        self.numerical_features = self.config.get("numerical_features", self._default_numerical_features)
        self.categorical_features = self.config.get("categorical_features", self._default_categorical_features)
        self.one_hot_features = self.config.get("one_hot_features", self._default_one_hot_features)
        self.id_columns = ["article_id", "week_num"]
        self.transformers = {}

    def process(
        self, transactions: pd.DataFrame, articles: pd.DataFrame, customers: pd.DataFrame
    ) -> ArticleDynamicFeatureResult:
        """Process transaction data to generate dynamic features for articles.

        Args:
            transactions: DataFrame containing transaction data
            articles: DataFrame containing article data
            customers: DataFrame containing customer data

        Returns:
            ArticleDynamicFeatureResult containing processed features
        """
        logger.info("Processing article dynamic features")

        # Ensure we have week_num column
        if "week_num" not in transactions.columns:
            raise ValueError("week_num column not found in transactions")

        # Extract transactions for the given time period
        transactions = transactions.query(
            "week_num >= @self.history_start_week_num and week_num <= @self.history_end_week_num"
        )

        # Generate a cross join of articles and weeks
        base_articles_week = self._generate_articles_week_cross_join(articles, self.start_week_num, self.end_week_num)

        # Calculate weekly statistics
        weekly_stats = self._calculate_weekly_statistics(transactions, customers)
        base_articles_week = self._generate_weekly_statistics_features(base_articles_week, weekly_stats)

        # Calculate cumulative mean age of customers who bought the article
        weekly_stats_age, customer_median_age = self._calculate_cumulative_stats(transactions, customers)
        base_articles_week = self._generate_cumulative_stats(base_articles_week, weekly_stats_age, customer_median_age)

        # Collect feature names
        self._collect_feature_names(base_articles_week)

        # Combine features
        return ArticleDynamicFeatureResult(
            data=base_articles_week,
            feature_names=self._get_feature_names_dict(),
        )

    def _generate_articles_week_cross_join(
        self, articles: pd.DataFrame, week_start_num: int, week_end_num: int
    ) -> pd.DataFrame:
        """Generate a cross join of articles and weeks."""
        logger.info(f"Generating a cross join of articles and weeks from {week_start_num} to {week_end_num}")
        articles_ids = articles.article_id.unique()
        logger.debug(f"Found {len(articles_ids)} unique articles")
        weeks = np.arange(week_start_num, week_end_num + 1)
        logger.debug(f"Found {len(weeks)} unique weeks")

        df1 = pd.DataFrame({"article_id": articles_ids})
        df2 = pd.DataFrame({"week_num": weeks})
        cross_join = pd.merge(df1, df2, how="cross")
        logger.debug(f"Cross join has shape: {cross_join.shape}")

        return cross_join

    def _calculate_weekly_statistics(self, transactions: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
        """Calculate weekly statistics for each article."""
        logger.info("Calculating weekly statistics")

        # Calculate sales count and revenue
        weekly_sales = (
            transactions.groupby(["article_id", "week_num"])
            .agg(
                weekly_sales_count=("t_dat", "count"),
                weekly_avg_price=("price", "mean"),
            )
            .reset_index()
        )
        logger.debug(f"Weekly sales has shape: {weekly_sales.shape}")
        logger.debug(f"Weekly sales: {weekly_sales.columns}")

        return weekly_sales

    def _generate_weekly_statistics_features(self, base: pd.DataFrame, weekly_sales: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating weekly statistics features")
        cols_key = ["article_id", "week_num"]
        cols_features = ["weekly_sales_count", "weekly_avg_price"]
        base = base.merge(weekly_sales[cols_key + cols_features], on=cols_key, how="left")

        # Fill in missing values for average price using ffill
        base.sort_values(cols_key, inplace=True)
        base["weekly_avg_price"] = base.groupby("article_id")["weekly_avg_price"].ffill()

        # For remaining missing prices, use average price of all products that week
        base["weekly_avg_price"] = base.groupby("week_num")["weekly_avg_price"].transform(lambda x: x.fillna(x.mean()))

        # For sales, use 0
        fillna_dict = {
            "weekly_sales_count": 0,
        }
        base.fillna(fillna_dict, inplace=True)
        logger.debug(f"Weekly statistics features have shape: {base.shape}")
        logger.debug(f"Weekly statistics features: {base.columns}")
        logger.debug(f"Missing values: {base.isna().sum()}")

        return base

    def _calculate_cumulative_stats(self, transactions: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
        """Calculate cumulative statistics of customers who bought the article."""
        logger.info("Calculating cumulative statistics of customers who bought the article")

        # Calculate cumulative mean age
        weekly_stats_age = transactions[["article_id", "week_num", "customer_id"]].merge(
            customers[["customer_id", "age"]], on="customer_id", how="left"
        )
        # Sort values for calculating cumulative sum and counts
        weekly_stats_age.sort_values(["week_num", "article_id", "customer_id"], inplace=True)

        # Calculate cumulative sum, count and mean of customer age
        weekly_stats_age["age_cum_sum"] = weekly_stats_age.groupby(["article_id"])["age"].cumsum()
        weekly_stats_age["age_cum_count"] = weekly_stats_age.groupby(["article_id"])["age"].cumcount() + 1
        weekly_stats_age["age_cum_mean"] = weekly_stats_age["age_cum_sum"] / weekly_stats_age["age_cum_count"]

        # Drop and rename columns
        weekly_stats_age.drop(columns=["age_cum_sum"], inplace=True)
        weekly_stats_age.rename(
            columns={"age_cum_mean": "cumulative_mean_age", "age_cum_count": "cumulative_sales_count"}, inplace=True
        )

        # Keep only the last row for each article-week
        weekly_stats_age = weekly_stats_age.groupby(["article_id", "week_num"]).last().reset_index()
        logger.debug(f"Cumulative mean age has shape: {weekly_stats_age.shape}")
        logger.debug(f"Cumulative mean age: {weekly_stats_age.columns}")

        # Get customer median age
        customer_median_age = customers.age.median()

        return weekly_stats_age, customer_median_age

    def _generate_cumulative_stats(
        self, base: pd.DataFrame, weekly_stats_age: pd.DataFrame, customer_median_age: float
    ) -> pd.DataFrame:
        logger.info("Generating cumulative statistics")
        # Merge base data with cumulative statistics
        cols_key = ["article_id", "week_num"]
        cols_features = ["cumulative_mean_age", "cumulative_sales_count"]
        base = base.merge(
            weekly_stats_age[cols_key + cols_features],
            on=cols_key,
            how="left",
        )

        # Fill NA using ffill
        base.sort_values(cols_key, inplace=True)
        base[cols_features] = base.groupby("article_id")[cols_features].ffill()

        # Fill in remaining missing values with median age or 0
        logger.debug("Filling in remaining missing values with median age or 0")
        fillna_dict = {
            "cumulative_mean_age": customer_median_age,
            "cumulative_sales_count": 0,
        }
        base.fillna(fillna_dict, inplace=True)

        logger.debug(f"Cumulative statistics have shape: {base.shape}")
        logger.debug(f"Cumulative statistics: {base.columns}")
        logger.debug(f"Missing values: {base.isna().sum()}")

        return base

    def _collect_feature_names(self, base: pd.DataFrame) -> None:
        """Collect feature names by type."""
        # Default classifications - override with config if needed
        # Only include columns that are actually present
        logger.debug(f"Collecting feature names for {len(base.columns)} columns")
        self.categorical_features = [col for col in self.categorical_features if col in base.columns]
        self.numerical_features = [col for col in self.numerical_features if col in base.columns]
        self.one_hot_features = [col for col in self.one_hot_features if col in base.columns]

        logger.debug(f"Collected {len(self.categorical_features)} categorical features")
        logger.debug(f"Collected {len(self.numerical_features)} numerical features")
        logger.debug(f"Collected {len(self.one_hot_features)} one-hot features")

    def _get_feature_names_dict(self) -> Dict[str, List[str]]:
        """Get feature names by type."""
        return {
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "one_hot_features": self.one_hot_features,
            "id_columns": self.id_columns,
        }


class ArticleDynamicFeaturePipeline:
    """End-to-end pipeline for processing article dynamic features."""

    def __init__(self, config: Optional[ArticleDynamicFeaturePipelineConfig] = None):
        """Initialize pipeline with config."""
        self.config = config or ArticleDynamicFeaturePipelineConfig()
        self.processor = None

    def setup(self, config: Optional[ArticleDynamicFeaturePipelineConfig] = None) -> "ArticleDynamicFeaturePipeline":
        """Set up the pipeline with configuration."""
        if config is not None:
            self.config = config

        logger.info("Setting up ArticleDynamicFeaturePipeline")
        logger.debug(f"Config: {json.dumps(self.config.to_dict(), indent=2)}")

        self.processor = ArticleDynamicFeatureProcessor(self.config.config_processor)
        return self

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load required data for processing."""
        logger.info("Loading data for ArticleDynamicFeaturePipeline")

        transactions = load_optimized_raw_data("transactions", subsample=self.config.subsample, seed=self.config.seed)
        articles = load_optimized_raw_data("articles", subsample=self.config.subsample, seed=self.config.seed)
        customers = load_optimized_raw_data("customers", subsample=self.config.subsample, seed=self.config.seed)

        return transactions, articles, customers

    def run(self) -> ArticleDynamicFeatureResult:
        """Run the pipeline to process article dynamic features."""
        if self.processor is None:
            raise ValueError("Pipeline not set up. Call setup() before running.")

        # Load data
        transactions, articles, customers = self._load_data()

        # Process features
        results = self.processor.process(transactions, articles, customers)

        # Save results
        path_to_dir = get_path_to_article_features("dynamic", self.config.subsample, self.config.seed)
        results.save(path_to_dir)

        return results


# ======================================================================================================================
# Article embeddings
# ======================================================================================================================
class TextDataset(Dataset):
    """For handling text data."""

    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class ImagePathDataset(Dataset):
    """Custom image class for loading articles image data and handle missing images."""

    def __init__(self, ids, transform=None, placeholder=None):
        self.ids = list(ids)
        self.transform = transform
        self.placeholder = placeholder

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = str(self.ids[idx])
        prefix = "0" + img_id[:2]
        img_path = f"../data/raw/images/{prefix}/0{img_id}.jpg"
        missing = 0
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        else:
            missing = 1
            if self.placeholder is not None:
                image = self.placeholder
            else:
                image = torch.zeros(3, 224, 224)
        return image, missing


@dataclass
class ArticleEmbeddingPipelineConfig:
    """Configuration for article embedding pipeline processing."""

    # Processor configuration
    config_processor: Dict[str, Any] = field(
        default_factory=lambda: {
            # Model configuration
            "text_model_id": "distilbert-base-uncased",
            "img_model_id": "resnet18",
            "batch_size": 32,
            "device_type": "mps",
            # Text features
            "cols_text": [
                "prod_name",
                "product_type_name",
                "product_group_name",
                "graphical_appearance_name",
                "colour_group_name",
                "perceived_colour_value_name",
                "perceived_colour_master_name",
                "department_name",
                "index_name",
                "index_group_name",
                "section_name",
                "garment_group_name",
                "detail_desc",
            ],
        }
    )

    # I/O Configuration
    subsample: float = 0.05
    seed: int = 42

    # Helper methods
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ArticleEmbeddingPipelineConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def create_default(cls) -> "ArticleEmbeddingPipelineConfig":
        """Create default configuration."""
        return cls.from_dict(DEFAULT_ARTICLE_EMBEDDING_CONFIG)


@dataclass
class ArticleEmbeddingResult:
    """Results from article embedding processing."""

    text_embeddings: np.ndarray
    image_embeddings: np.ndarray
    image_missing: np.ndarray
    id_to_index: Dict[int, int]
    index_to_id: Dict[int, int]

    def save(self, path_to_dir: str):
        """Save result to disk.

        Args:
            path_to_dir: Directory path to save the results
        """
        logger.info(f"Saving article embedding result to {path_to_dir}")
        os.makedirs(path_to_dir, exist_ok=True)

        # Save components separately
        np.save(f"{path_to_dir}/text_embeddings.npy", self.text_embeddings)
        logger.info(f"Saved text embeddings to {path_to_dir}/text_embeddings.npy")

        np.save(f"{path_to_dir}/image_embeddings.npy", self.image_embeddings)
        logger.info(f"Saved image embeddings to {path_to_dir}/image_embeddings.npy")

        np.save(f"{path_to_dir}/image_missing.npy", self.image_missing)
        logger.info(f"Saved image missing flags to {path_to_dir}/image_missing.npy")

        with open(f"{path_to_dir}/id_to_index.json", "w") as f:
            json.dump(self.id_to_index, f)
        logger.info(f"Saved id to index mapping to {path_to_dir}/id_to_index.json")

        with open(f"{path_to_dir}/index_to_id.json", "w") as f:
            json.dump(self.index_to_id, f)
        logger.info(f"Saved index to id mapping to {path_to_dir}/index_to_id.json")

    @classmethod
    def load(cls, path_to_dir: str) -> "ArticleEmbeddingResult":
        """Load result from disk.

        Args:
            path_to_dir: Directory path containing the saved results

        Returns:
            ArticleEmbeddingResult containing loaded data
        """
        logger.info(f"Loading article embedding result from {path_to_dir}")

        # Load components
        text_embeddings = np.load(f"{path_to_dir}/text_embeddings.npy")
        logger.info(f"Loaded text embeddings from {path_to_dir}/text_embeddings.npy")

        image_embeddings = np.load(f"{path_to_dir}/image_embeddings.npy")
        logger.info(f"Loaded image embeddings from {path_to_dir}/image_embeddings.npy")

        image_missing = np.load(f"{path_to_dir}/image_missing.npy")
        logger.info(f"Loaded image missing flags from {path_to_dir}/image_missing.npy")

        with open(f"{path_to_dir}/id_to_index.json", "r") as f:
            id_to_index = json.load(f)
        id_to_index = {int(k): v for k, v in id_to_index.items()}
        logger.info(f"Loaded id to index mapping from {path_to_dir}/id_to_index.json")

        # FIX: This is temporary since this is missing from the initial run
        if os.path.exists(f"{path_to_dir}/index_to_id.json"):
            with open(f"{path_to_dir}/index_to_id.json", "r") as f:
                index_to_id = json.load(f)
            index_to_id = {int(k): v for k, v in index_to_id.items()}
            logger.info(f"Loaded index to id mapping from {path_to_dir}/index_to_id.json")
        else:
            index_to_id = {idx: id_ for id_, idx in id_to_index.items()}

        return cls(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            image_missing=image_missing,
            id_to_index=id_to_index,
            index_to_id=index_to_id,
        )


class ArticleEmbeddingProcessor:
    """Process article data to generate text and image embeddings."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize processor with configuration."""
        self.config = config
        if config.get("device_type") is not None:
            self.device = torch.device(config.get("device_type"))
        else:
            self.device = get_device()
        logger.info(f"Using device: {self.device}")

        # Model configuration
        self._default_text_model_id = "distilbert-base-uncased"
        self._default_img_model_name = "resnet18"
        self.text_model_id = config.get("text_model_id", self._default_text_model_id)
        self.img_model_name = config.get("img_model_id", self._default_img_model_name)

        # Load pre-trained models
        self.text_model, self.tokenizer = self._load_pretrained_model_text()
        self.img_model = self._load_pretrained_model_img()
        self.batch_size = config.get("batch_size", 32)

        # Text features configuration
        self._default_cols_text = [
            "prod_name",
            "product_type_name",
            "product_group_name",
            "graphical_appearance_name",
            "colour_group_name",
            "perceived_colour_value_name",
            "perceived_colour_master_name",
            "department_name",
            "index_name",
            "index_group_name",
            "section_name",
            "garment_group_name",
            "detail_desc",
        ]
        self.cols_text = config.get("cols_text", self._default_cols_text)

    def process(self, articles: pd.DataFrame) -> ArticleEmbeddingResult:
        """Process the articles dataframe to generate embeddings.

        Args:
            articles: DataFrame containing article data

        Returns:
            ArticleEmbeddingResult containing text and image embeddings
        """
        logger.info("Processing article embeddings")

        # Validate input data
        if not all(col in articles.columns for col in self.cols_text):
            raise ValueError(f"Columns {self.cols_text} not found in articles")

        # Generate text embeddings
        logger.info("Generating text embeddings")
        text_embeddings = self._generate_text_embeddings(articles)

        # Generate image embeddings
        logger.info("Generating image embeddings")
        image_embeddings, image_missing = self._generate_image_embeddings(articles.article_id.tolist())

        # Create id to index mapping
        id_to_index = {id_: idx for idx, id_ in enumerate(articles.article_id)}
        index_to_id = {idx: id_ for id_, idx in enumerate(articles.article_id)}

        return ArticleEmbeddingResult(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            image_missing=image_missing,
            id_to_index=id_to_index,
            index_to_id=index_to_id,
        )

    def _load_pretrained_model_text(self) -> Tuple[torch.nn.Module, Any]:
        """Load pretrained model for text embeddings.

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading text model: {self.text_model_id}")
        if self.text_model_id == "distilbert-base-uncased":
            tokenizer = DistilBertTokenizer.from_pretrained(self.text_model_id)
            model = DistilBertModel.from_pretrained(self.text_model_id).to(self.device)
        else:
            raise ValueError(f"Model {self.text_model_id} not supported")
        logger.info(f"Loaded text model: {self.text_model_id}")
        return model, tokenizer

    def _load_pretrained_model_img(self) -> torch.nn.Module:
        """Load a pre-trained model for image embeddings.

        Returns:
            Pre-trained image model
        """
        logger.info(f"Loading image model: {self.img_model_name}")
        if self.img_model_name == "resnet18":
            model = models.resnet18(pretrained=True)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model = model.to(self.device)
            model.eval()
        elif self.img_model_name == "vit_b_16":
            model = models.vit_b_16(pretrained=True)
            model.heads = torch.nn.Identity()  # Remove classification head
            model = model.to(self.device)
            model.eval()
        else:
            raise ValueError(f"Model {self.img_model_name} not supported")
        logger.info(f"Loaded image model: {self.img_model_name}")
        return model

    def _generate_text_embeddings(self, articles: pd.DataFrame) -> np.ndarray:
        """Generate text embeddings for the articles dataframe.

        Args:
            articles: DataFrame containing article data

        Returns:
            Array of text embeddings
        """
        logger.info("Generating text embeddings")
        # Load data into dataloader
        texts = concatenate_text_columns(articles, self.cols_text)
        dataset = TextDataset(texts)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Generate embeddings in batches
        embeddings = []
        for batch_texts in tqdm(loader, desc="Processing text batches"):
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(
                self.device
            )

            with torch.no_grad():
                outputs = self.text_model(**inputs)
                # Generate the average text embedding excluding the [CLS] token
                batch_emb = outputs.last_hidden_state[:, 1:, :].mean(dim=1).cpu()
                embeddings.extend(batch_emb)

        logger.info(f"Generated {len(embeddings)} text embeddings")
        logger.info(f"Text embeddings shape: {np.array(embeddings).shape}")

        return np.array(embeddings)

    def _generate_image_embeddings(self, article_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate image embeddings for the articles.

        Args:
            article_ids: List of article IDs

        Returns:
            Tuple of (image embeddings array, missing images array)
        """
        logger.info("Generating image embeddings")
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Set up dataloader
        logger.info(f"Setting up dataloader with batch size: {self.batch_size}")
        dataset = ImagePathDataset(article_ids, transform=transform)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        embeddings = []
        missing = []

        # Generate embeddings in batches
        for batch_images, batch_missing in tqdm(loader, desc="Processing image batches"):
            batch_images = batch_images.to(self.device)
            with torch.no_grad():
                outputs = self.img_model(batch_images).reshape(-1, 512)  # ResNet18
                batch_emb = outputs.cpu()
                embeddings.extend(batch_emb)
                missing.extend(batch_missing)

        logger.info(f"Generated {len(embeddings)} image embeddings")
        logger.info(f"Image embeddings shape: {np.array(embeddings).shape}")
        logger.info(f"Number of missing images: {np.sum(np.array(missing))}")

        return np.array(embeddings), np.array(missing)


class ArticleEmbeddingPipeline:
    """End-to-end pipeline for processing article embeddings."""

    def __init__(self, config: Optional[ArticleEmbeddingPipelineConfig] = None):
        """Initialize pipeline with config."""
        self.config = config or ArticleEmbeddingPipelineConfig()
        self.processor = None

    def setup(self, config: Optional[ArticleEmbeddingPipelineConfig] = None) -> "ArticleEmbeddingPipeline":
        """Set up the pipeline with configuration.

        Args:
            config: Optional configuration to override defaults

        Returns:
            Self for method chaining
        """
        if config is not None:
            self.config = config

        logger.info("Setting up ArticleEmbeddingPipeline")
        logger.debug(f"Config: {json.dumps(self.config.to_dict(), indent=2)}")

        self.processor = ArticleEmbeddingProcessor(self.config.config_processor)
        return self

    def _load_data(self) -> pd.DataFrame:
        """Load required data for processing.

        Returns:
            DataFrame containing article data
        """
        logger.info("Loading data for ArticleEmbeddingPipeline")
        articles = load_optimized_raw_data("articles", subsample=self.config.subsample, seed=self.config.seed)
        logger.debug(f"Loaded article data with shape: {articles.shape}")
        return articles

    def run(self) -> ArticleEmbeddingResult:
        """Run the pipeline to process article embeddings.

        Returns:
            ArticleEmbeddingResult containing processed embeddings

        Raises:
            ValueError: If pipeline is not set up
        """
        if self.processor is None:
            raise ValueError("Pipeline not set up. Call setup() before running.")

        # Load data
        articles = self._load_data()

        # Process embeddings
        results = self.processor.process(articles)

        # Save results
        path_to_dir = get_path_to_article_features("embedding", self.config.subsample, self.config.seed)
        results.save(path_to_dir)

        return results
