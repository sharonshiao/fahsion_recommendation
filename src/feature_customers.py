import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.config import (
    DEFAULT_CUSTOMER_DYNAMIC_FEATURES_CONFIG,
    DEFAULT_CUSTOMER_STATIC_FEATURES_CONFIG,
)
from src.feature_extraction import load_optimized_raw_data
from src.features_articles import (
    ArticleEmbeddingResult,
    get_path_to_article_features,
)

logger = logging.getLogger(__name__)

CUSTOMERS_FEATURE_TYPES = ["static", "dynamic"]


def get_path_to_customers_features(feature_type: str, subsample: float, seed: int) -> str:
    """Get path to customers features."""
    if feature_type not in CUSTOMERS_FEATURE_TYPES:
        raise ValueError(f"Invalid feature type: {feature_type}. Supported types: {CUSTOMERS_FEATURE_TYPES}")
    if subsample < 1:
        return f"../data/preprocessed/customers_{feature_type}/subsample_{subsample}_{seed}"
    else:
        return f"../data/preprocessed/customers_{feature_type}/full"


# ======================================================================================================================
# Customer features
# =====================================================================================================================
@dataclass
class CustomerFeaturePipelineConfig:
    """Configuration for customer feature pipeline processing."""

    config_processor: Dict[str, Any] = field(
        default_factory=lambda: {
            # Feature encoding
            "encoding_strategy": "ordinal",  # Options: None, "ordinal", "one_hot"
            # Feature classification
            "categorical_features": ["club_member_status", "fashion_news_frequency", "postal_code", "age_bin"],
            "numerical_features": ["fn", "active", "age"],
            "one_hot_features": [],
            "missing_value_strategy": "fill_unknown",
            "missing_values_map": {
                "fn": 0,
                "active": 0,
                "club_member_status": "unknown",
                "fashion_news_frequency": "unknown",
                "postal_code": "unknown",
            },
            "age_bins": [-np.inf, 18, 25, 35, 45, 55, 65, np.inf],
            "keep_numeric_age": False,
        }
    )

    # I/O Configuration
    subsample: float = 1
    seed: int = 42

    # Helper methods
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CustomerFeaturePipelineConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def create_default(cls) -> "CustomerFeaturePipelineConfig":
        """Create default configuration."""
        return cls.from_dict(DEFAULT_CUSTOMER_STATIC_FEATURES_CONFIG)


@dataclass
class CustomerFeatureResult:
    """Results from customer feature processing."""

    data: pd.DataFrame  # Processed customer dataframe
    feature_names: Dict[str, List[str]]  # Features by type
    transformers: Dict[str, any]  # Encoding transformers

    def save(self, path_to_dir: str):
        """Save result to disk."""
        logger.info(f"Saving customer feature result to {path_to_dir}")
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
        logger.info(f"Loading customer feature result from {path_to_dir}")

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


class CustomerStaticFeatureProcessor:
    """Process customer data and generate features in a pipeline fashion."""

    def __init__(self, config=None):
        self.config = config or {}
        self.encoding_strategy = self.config.get("encoding_strategy", "ordinal")
        self.age_bins = self.config.get("age_bins", [-np.inf, 18, 25, 35, 45, 55, 65, np.inf])
        self.keep_numeric_age = self.config.get("keep_numeric_age", False)
        self.missing_value_strategy = self.config.get("missing_value_strategy", "fill_unknown")
        self.missing_values_map = self.config.get(
            "missing_values_map",
            {
                "fn": 0,
                "active": 0,
                "club_member_status": "unknown",
                "fashion_news_frequency": "unknown",
                "postal_code": "unknown",
            },
        )

        self._default_categorical_features = ["club_member_status", "fashion_news_frequency", "postal_code", "age_bin"]
        self._default_numerical_features = ["fn", "active", "age"]
        self._default_one_hot_features = []

        self.categorical_features = self.config.get("categorical_features", self._default_categorical_features)
        self.numerical_features = self.config.get("numerical_features", self._default_numerical_features)
        self.one_hot_features = self.config.get("one_hot_features", self._default_one_hot_features)
        self.transformers = {}

    def process(self, customers_raw: pd.DataFrame) -> dict:
        """
        Main method to process customer data end-to-end.

        Args:
            customers_raw: Raw customer DataFrame

        Returns:
            Dict containing processed data and feature metadata
        """
        logger.info("Processing customer data")
        customers = customers_raw.copy()

        # Apply preprocessing steps in sequence
        customers = self._standardize_column_names(customers)
        customers = self._handle_missing_values(customers)
        customers = self._create_demographic_features(customers)
        customers = self._encode_categorical_features(customers)

        # Collect feature names
        self._collect_feature_names(customers)
        results = CustomerFeatureResult(
            data=customers,
            feature_names={
                "numerical_features": self.numerical_features,
                "categorical_features": self.categorical_features,
                "one_hot_features": self.one_hot_features,
                "id_columns": self.id_columns,
            },
            transformers=self.transformers,
        )
        return results

    def _standardize_column_names(self, customers: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase."""
        logger.debug("Standardizing column names")
        customers.columns = customers.columns.str.lower()
        return customers

    def _handle_missing_values(self, customers: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in customer data."""
        logger.debug("Handling missing values")

        if self.missing_value_strategy == "fill_unknown":
            customers.fillna(self.missing_values_map, inplace=True)
        elif self.missing_value_strategy == "drop":
            customers.dropna(inplace=True)
        else:
            raise ValueError(f"Invalid missing value strategy: {self.missing_value_strategy}")

        return customers

    def _create_demographic_features(self, customers: pd.DataFrame) -> pd.DataFrame:
        """Create demographic features."""
        logger.debug("Creating demographic features")

        # Process age
        logger.info("Processing age feature")
        customers["age_bin"] = pd.cut(customers["age"], bins=self.age_bins, labels=False)
        customers["age_bin"] = customers["age_bin"].astype(str)

        # Optionally keep numeric age
        if not self.keep_numeric_age:
            customers.drop("age", axis=1, inplace=True)

        return customers

    def _encode_categorical_features(self, customers: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features if specified in config."""

        if self.encoding_strategy == "ordinal":
            logger.debug("Applying ordinal encoding")

        for col in self.categorical_features:
            if col in customers.columns:
                oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                customers[col] = oe.fit_transform(customers[col].astype(str).values.reshape(-1, 1))
                self.transformers[f"ordinal_{col}"] = oe

        return customers

    def _collect_feature_names(self, customers: pd.DataFrame) -> None:
        """Collect feature names by type."""
        logger.info(f"Collecting feature names for {len(customers.columns)} columns")
        # Only include columns that are actually present
        self.categorical_features = [col for col in self.categorical_features if col in customers.columns]
        self.numerical_features = [col for col in self.numerical_features if col in customers.columns]
        self.one_hot_features = [col for col in self.one_hot_features if col in customers.columns]

        logger.debug(f"Collected {len(self.categorical_features)} categorical features")
        logger.debug(f"Collected {len(self.numerical_features)} numerical features")
        logger.debug(f"Collected {len(self.one_hot_features)} one-hot features")

    @property
    def id_columns(self) -> str:
        """Return the ID column name."""
        return ["customer_id"]


class CustomerFeaturePipeline:
    """End-to-end pipeline for processing customer features."""

    def __init__(self, config=None):
        """Initialize pipeline with config."""
        self.config = config or {}
        self.processor = None

    def setup(self, config=None):
        """Set up the pipeline with configuration."""
        if config is not None:
            self.config = config

        logger.info("Setting up CustomerFeaturePipeline with config:")
        logger.debug(json.dumps(asdict(self.config), indent=2))
        self.processor = CustomerStaticFeatureProcessor(self.config.config_processor)
        return self

    def _load_raw_data(self):
        logger.info(f"Loading raw customer data for CustomerFeaturePipeline")

        customers = load_optimized_raw_data("customers", subsample=self.config.subsample, seed=self.config.seed)
        logger.debug(f"Loaded raw customer data with shape: {customers.shape}")
        return customers

    def run(self):

        if self.processor is None:
            raise ValueError("Pipeline not set up. Call setup() before running the pipeline.")

        customers_raw = self._load_raw_data()

        # Process data
        results = self.processor.process(customers_raw)

        # Save results
        path_to_dir = get_path_to_customers_features("static", self.config.subsample, self.config.seed)
        results.save(path_to_dir)

        return results


# ======================================================================================================================
# Customer dynamic features
# ======================================================================================================================


@dataclass
class CustomerDynamicFeaturePipelineConfig:
    """Configuration for customer dynamic feature pipeline processing."""

    # Feature configuration
    config_processor: Dict[str, Any] = field(
        default_factory=lambda: {
            # Feature configuration
            "start_week_num": 76,
            "end_week_num": 102,
            "k_items": 5,
        }
    )

    # I/O Configuration
    subsample: float = 0.05
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CustomerDynamicFeaturePipelineConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def create_default(cls) -> "CustomerDynamicFeaturePipelineConfig":
        """Create default configuration."""
        return cls.from_dict(DEFAULT_CUSTOMER_DYNAMIC_FEATURES_CONFIG)


@dataclass
class CustomerDynamicFeatureResult:
    """Results from customer dynamic feature processing."""

    data: pd.DataFrame  # Processed customer dynamic features
    feature_names: Dict[str, List[str]]  # Features by type

    def save(self, path_to_dir: str):
        """Save result to disk."""
        logger.info(f"Saving customer dynamic feature result to {path_to_dir}")
        # Create directory if it doesn't exist already
        os.makedirs(path_to_dir, exist_ok=True)

        # Save components separately
        self.data.to_parquet(f"{path_to_dir}/data.parquet")
        logger.info(f"Saved data to {path_to_dir}/data.parquet")

        with open(f"{path_to_dir}/feature_names.json", "w") as f:
            json.dump(self.feature_names, f)
        logger.info(f"Saved feature names to {path_to_dir}/feature_names.json")

    @classmethod
    def load(cls, path_to_dir: str):
        """Load result from disk."""
        logger.info(f"Loading customer dynamic feature result from {path_to_dir}")

        # Load components
        data = pd.read_parquet(f"{path_to_dir}/data.parquet")
        logger.info(f"Loaded data from {path_to_dir}/data.parquet")

        with open(f"{path_to_dir}/feature_names.json", "r") as f:
            feature_names = json.load(f)
        logger.info(f"Loaded feature names from {path_to_dir}/feature_names.json")

        return cls(data=data, feature_names=feature_names)

    def get_feature_list(self, include_id: bool = True) -> List[str]:
        """Get a flattened list of all features."""
        features = []
        for feature_type, feature_list in self.feature_names.items():
            if not include_id and feature_type == "id_columns":
                continue
            features.extend(feature_list)

        return features


class CustomerDynamicFeatureProcessor:
    """Process customer dynamic features."""

    def __init__(self, config=None):
        self.config = config or {}
        self.start_week_num = self.config.get("start_week_num", 76)
        self.end_week_num = self.config.get("end_week_num", 102)
        self.k_items = self.config.get("k_items", 5)
        self.id_columns = ["customer_id", "week_num"]
        self.numerical_features = ["customer_avg_price"]
        self.embedding_features = ["customer_avg_text_embedding", "customer_avg_image_embedding"]
        self.one_hot_features = []
        self.categorical_features = []

    def process(self, raw_transactions: pd.DataFrame, article_embeddings: ArticleEmbeddingResult) -> pd.DataFrame:
        """Process customer dynamic features.

        Args:
            transactions: DataFrame containing transaction data
            article_embeddings: ArticleEmbeddingResult object containing text and image embeddings

        Returns:
            DataFrame with average embeddings for each customer-week pair
        """
        logger.info("Processing customer dynamic features")

        # Filter transactions to only include weeks between start_week_num and end_week_num
        transactions = raw_transactions[
            (raw_transactions["week_num"] >= self.start_week_num) & (raw_transactions["week_num"] <= self.end_week_num)
        ]

        # Get last k items by week
        last_k_items = self._get_last_k_items_by_week(transactions, self.k_items)

        # Calculate average embeddings
        customer_week_avg_embeddings = self._calculate_average_embeddings(last_k_items, article_embeddings)

        # Generate cross join of all possible customer-week pairs
        cross_join = self._generate_customer_week_cross_join(customer_week_avg_embeddings)

        # Join with average embeddings
        cross_join = cross_join.merge(customer_week_avg_embeddings, on=["customer_id", "week_num"], how="left")
        logger.debug(f"Cross join has shape: {cross_join.shape}")
        logger.debug(f"Cross join has columns: {cross_join.columns}")

        # Fill missing values
        cross_join = self._fill_missing_values(cross_join)

        return CustomerDynamicFeatureResult(
            data=cross_join,
            feature_names=self._get_feature_names_dict(),
        )

    def _get_last_k_items_by_week(self, transactions_df: pd.DataFrame, k_items: int) -> pd.DataFrame:
        """Get the last k items purchased by each customer in each week.

        Args:
            transactions_df: DataFrame containing transaction data

        Returns:
            DataFrame with last k items for each customer-week pair
        """
        logger.info(f"Getting last {k_items} items by week")
        # Sort by date (most recent first) within each customer and week
        # Also sort by article_id to maintain reproducibility
        sorted_transactions = transactions_df.sort_values(
            ["customer_id", "week_num", "t_dat", "article_id"], ascending=[True, False, False, True]
        )

        # Group by customer and week, and get the last k items
        last_k_items = (
            sorted_transactions.groupby(["customer_id", "week_num"])
            .agg(
                {"article_id": lambda x: list(x)[:k_items], "price": "mean"}
            )  # Get last k items (since we sorted in descending order)
            .reset_index()
        )
        logger.debug(f"Last {k_items} items by week has shape: {last_k_items.shape}")

        return last_k_items

    def _calculate_average_embeddings(
        self, last_k_items_df: pd.DataFrame, article_embeddings: ArticleEmbeddingResult
    ) -> pd.DataFrame:
        """Calculate average embeddings for each customer-week combination.

        Args:
            last_k_items_df: DataFrame with columns ['customer_id', 'week_num', 'article_id']
            article_embeddings: ArticleEmbeddingResult object containing text and image embeddings

        Returns:
            DataFrame with average embeddings for each customer-week
        """
        logger.info(f"Calculating average embeddings for {len(last_k_items_df)} customer-week combinations")
        # Create a list to store results
        results = []

        # Process each customer-week combination
        for _, row in last_k_items_df.iterrows():
            customer_id = row["customer_id"]
            week_num = row["week_num"]
            article_ids = row["article_id"]
            avg_price = row["price"]

            # Get embeddings for each article
            text_embeddings_list = []
            image_embeddings_list = []
            for article_id in article_ids:
                if article_id in article_embeddings.id_to_index:
                    idx = article_embeddings.id_to_index[article_id]
                    # Combine text and image embeddings
                    text_emb = article_embeddings.text_embeddings[idx]
                    img_emb = article_embeddings.image_embeddings[idx]
                    text_embeddings_list.append(text_emb)
                    image_embeddings_list.append(img_emb)

            if text_embeddings_list:
                # Calculate average embedding
                avg_text_embedding = np.mean(text_embeddings_list, axis=0)
                avg_image_embedding = np.mean(image_embeddings_list, axis=0)
                results.append(
                    {
                        "customer_id": customer_id,
                        "week_num": week_num,
                        "customer_avg_price": avg_price,
                        "customer_avg_text_embedding": avg_text_embedding,
                        "customer_avg_image_embedding": avg_image_embedding,
                    }
                )

        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        logger.debug(f"Results DataFrame has shape: {results_df.shape}")

        return results_df

    def _generate_customer_week_cross_join(self, customer_week_avg_embeddings: pd.DataFrame) -> pd.DataFrame:
        """Generate a cross join of all possible customer-week pairs."""
        logger.info("Generating cross join of customer-week pairs")

        # Get unique customer IDs and weeks
        customer_ids = customer_week_avg_embeddings["customer_id"].unique()
        weeks = np.arange(self.start_week_num, self.end_week_num + 1)

        # Create cross join
        df1 = pd.DataFrame(customer_ids, columns=["customer_id"])
        df2 = pd.DataFrame(weeks, columns=["week_num"])
        cross_join = pd.merge(df1, df2, how="cross")

        logger.debug(f"Cross join has shape: {cross_join.shape}")
        logger.debug(f"Cross join has columns: {cross_join.columns}")
        return cross_join

    def _fill_missing_values(self, cross_join: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in the cross join DataFrame.

        Args:
            cross_join: DataFrame with customer-week pairs

        Returns:
            DataFrame with filled missing values

        Note:
            We use forward fill to fill missing values. We do not fill missing values with 0 numpy array to save memory,
        """
        logger.info("Filling missing values")

        # Sort by customer_id and week_num for forward fill
        cross_join = cross_join.sort_values(["customer_id", "week_num"]).reset_index(drop=True)

        # Forward fill missing values within each customer group
        cols_ffill = ["customer_avg_price", "customer_avg_text_embedding", "customer_avg_image_embedding"]
        cross_join[cols_ffill] = cross_join.groupby("customer_id")[cols_ffill].ffill()

        # Use weekly mean price to fill missing values
        cross_join["customer_avg_price"] = cross_join.groupby("week_num")["customer_avg_price"].transform("mean")

        logger.debug(f"Cross join after filling missing values has shape: {cross_join.shape}")
        logger.debug(f"Cross join after filling missing values has columns: {cross_join.columns}")
        logger.debug(f"Missing values filled: {cross_join.isna().sum()}")
        return cross_join

    def _get_feature_names_dict(self) -> Dict[str, List[str]]:
        """Get feature names by type."""
        return {
            "numerical_features": self.numerical_features,
            "embedding_features": self.embedding_features,
            "one_hot_features": self.one_hot_features,
            "categorical_features": self.categorical_features,
            "id_columns": self.id_columns,
        }


class CustomerDynamicFeaturePipeline:
    """End-to-end pipeline for processing customer dynamic features.


    This pipeline will generate the features for all customer-weeks
    """

    def __init__(self, config: Optional[CustomerDynamicFeaturePipelineConfig] = None):
        """Initialize pipeline with config."""
        self.config = config or CustomerDynamicFeaturePipelineConfig()
        self.processor = None

    def setup(self, config: Optional[CustomerDynamicFeaturePipelineConfig] = None) -> "CustomerDynamicFeaturePipeline":
        """Set up the pipeline with configuration."""
        if config is not None:
            self.config = config

        logger.info("Setting up CustomerDynamicFeaturePipeline")
        logger.debug(f"Config: {json.dumps(self.config.to_dict(), indent=2)}")

        self.processor = CustomerDynamicFeatureProcessor(self.config.config_processor)
        return self

    def _load_data(self) -> Tuple[pd.DataFrame, ArticleEmbeddingResult]:
        """Load required data for processing."""
        logger.info("Loading data for CustomerDynamicFeaturePipeline")

        transactions_train = load_optimized_raw_data(
            data_type="transactions", sample="train", subsample=self.config.subsample, seed=self.config.seed
        )
        transactions_valid = load_optimized_raw_data(
            data_type="transactions", sample="valid", subsample=self.config.subsample, seed=self.config.seed
        )
        transactions_test = load_optimized_raw_data(
            data_type="transactions", sample="test", subsample=self.config.subsample, seed=self.config.seed
        )
        transactions = pd.concat([transactions_train, transactions_valid, transactions_test])

        # Load article embeddings
        # We use the full embedding
        path_article_embedding = get_path_to_article_features("embedding", subsample=1, seed=self.config.seed)
        article_embeddings = ArticleEmbeddingResult.load(path_to_dir=path_article_embedding)

        return transactions, article_embeddings

    def run(self) -> CustomerDynamicFeatureResult:
        """Run the pipeline to process customer dynamic features."""
        if self.processor is None:
            raise ValueError("Pipeline not set up. Call setup() before running.")

        # Load data
        transactions, article_embeddings = self._load_data()

        # Process features
        results = self.processor.process(transactions, article_embeddings)

        # Save results
        path_to_dir = get_path_to_customers_features("dynamic", self.config.subsample, self.config.seed)
        results.save(path_to_dir)

        return results
