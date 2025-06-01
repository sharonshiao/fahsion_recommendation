import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from src.candidate_generator import (
    CandidateGeneratorResult,
    get_path_to_candidates,
)
from src.config import (
    DEFAULT_LIGHTGBM_DATA_PROCESSOR_TEST_CONFIG,
    DEFAULT_LIGHTGBM_DATA_PROCESSOR_TRAIN_CONFIG,
    DEFAULT_LIGHTGBM_DATA_PROCESSOR_VALID_CONFIG,
)
from src.feature_customers import (
    CustomerDynamicFeatureResult,
    CustomerStaticFeatureResult,
    get_path_to_customers_features,
)
from src.features_articles import (
    ArticleDynamicFeatureResult,
    ArticleEmbeddingResult,
    ArticleStaticFeatureResult,
    get_path_to_article_features,
)
from src.utils.embeddings import calculate_df_batch_cosine_similarity

logger = logging.getLogger(__name__)


def get_path_to_lightgbm_data(sample: str, use_type: str, subsample: float, seed: int) -> str:
    """Get path to LightGBM data."""
    if use_type == "train":
        path = f"../data/model/input/{sample}"
    elif use_type == "inference":
        path = f"../data/model/input_inference/{sample}"
    else:
        raise ValueError(f"Invalid use type: {use_type}")

    if subsample < 1:
        path += f"/subsample_{subsample}_{seed}"
    else:
        path += "/full"
    return path


# ======================================================================================================================
# LightGBM data
# ======================================================================================================================
@dataclass
class LightGBMDataProcessorConfig:
    """Configuration for LightGBM data preprocessing pipeline."""

    # Sample
    sample: str = "train"  # Sample type ('train', 'valid', 'test')

    # Feature selection options
    include_article_static_features: bool = True
    include_article_dynamic_features: bool = True
    include_customer_static_features: bool = True
    include_transaction_features: bool = True
    include_user_history: bool = False

    # I/O
    use_default_data_paths: bool = True  # Use default paths for data
    subsample: float = 0.05  # Subsample rate for training data
    seed: int = 42  # Random seed for reproducibility
    data_paths: Dict[str, str] = field(
        default_factory=lambda: {"candidates": "", "article_features": "", "customer_features": ""}
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def create_default(cls, sample="train") -> "LightGBMDataProcessorConfig":
        """Create default configuration."""
        if sample == "train":
            return cls(**DEFAULT_LIGHTGBM_DATA_PROCESSOR_TRAIN_CONFIG)
        elif sample == "valid":
            return cls(**DEFAULT_LIGHTGBM_DATA_PROCESSOR_VALID_CONFIG)
        elif sample == "test":
            return cls(**DEFAULT_LIGHTGBM_DATA_PROCESSOR_TEST_CONFIG)
        else:
            raise NotImplementedError


@dataclass
class LightGBMDataResult:
    """Results from LightGBM data preprocessing."""

    data: pd.DataFrame  # Training data
    label: Optional[np.ndarray] = None  # Training labels
    group: Optional[np.ndarray] = None  # Groups for ranking
    use_type: str = "train"  # Use type ('train', 'inference')
    feature_names: Dict[str, List[str]] = field(default_factory=dict)  # Feature names by type
    transformers: Dict[str, Any] = field(default_factory=dict)  # Any fitted transformers
    sample: str = "train"  # Sample type ('train', 'valid', 'test')
    default_prediction: Optional[np.ndarray] = None  # Default predictions (if any)

    def save(self, path_to_dir: str):
        """Save result to disk."""

        # Create directory if it doesn't exist
        os.makedirs(path_to_dir, exist_ok=True)

        # Save dataframes
        self.data.to_parquet(f"{path_to_dir}/data.parquet")
        logger.info(f"Saved data to {path_to_dir}/data.parquet")

        # Save numpy arrays
        if self.label is not None:
            np.save(f"{path_to_dir}/label.npy", self.label)
            logger.info(f"Saved label to {path_to_dir}/label.npy")
        if self.group is not None:
            np.save(f"{path_to_dir}/group.npy", self.group)
            logger.info(f"Saved group to {path_to_dir}/group.npy")

        # Save feature names and transformers
        with open(f"{path_to_dir}/feature_names.json", "w") as f:
            json.dump(self.feature_names, f)
        logger.info(f"Saved feature names to {path_to_dir}/feature_names.json")

        with open(f"{path_to_dir}/transformers.pkl", "wb") as f:
            pickle.dump(self.transformers, f)
        logger.info(f"Saved transformers to {path_to_dir}/transformers.pkl")

        if self.default_prediction is not None:
            np.save(f"{path_to_dir}/default_prediction.npy", self.default_prediction)
            logger.info(f"Saved default prediction to {path_to_dir}/default_prediction.npy")

        # Save meta data
        metadata = {
            "has_label": self.label is not None,
            "has_group": self.group is not None,
            "use_type": self.use_type,
            "sample": self.sample,
            "has_default_prediction": self.default_prediction is not None,
        }
        with open(f"{path_to_dir}/metadata.json", "w") as f:
            json.dump(metadata, f)
            logger.info(f"LightGBMDataResult metadata saved to {path_to_dir}/metadata.json")

    @classmethod
    def load(cls, path_to_dir: str):
        """Load result from disk."""

        # Load dataframes if they exist
        data = pd.read_parquet(f"{path_to_dir}/data.parquet")

        # Read metadata to check if label and group exist
        metadata_path = f"{path_to_dir}/metadata.json"
        with open(f"{path_to_dir}/metadata.json", "r") as f:
            metadata = json.load(f)
        logger.info(f"Metadata loaded: {metadata}")

        # Load numpy arrays if they exist
        label = None
        if metadata["has_label"]:
            label = np.load(f"{path_to_dir}/label.npy")
            logger.info(f"Loaded label from {path_to_dir}/label.npy")
        group = None
        if metadata["has_group"]:
            group = np.load(f"{path_to_dir}/group.npy")
            logger.info(f"Loaded group from {path_to_dir}/group.npy")

        default_prediction = None
        if metadata["has_default_prediction"]:
            default_prediction = np.load(f"{path_to_dir}/default_prediction.npy")
            logger.info(f"Loaded default prediction from {path_to_dir}/default_prediction.npy")

        sample = metadata["sample"]

        # Load feature names and transformers
        with open(f"{path_to_dir}/feature_names.json", "r") as f:
            feature_names = json.load(f)
        with open(f"{path_to_dir}/transformers.pkl", "rb") as f:
            transformers = pickle.load(f)

        return cls(
            data=data,
            label=label,
            group=group,
            use_type=metadata["use_type"],
            feature_names=feature_names,
            transformers=transformers,
            sample=sample,
            default_prediction=default_prediction,
        )

    def get_feature_names_list(self) -> List[str]:
        """Get feature names."""
        col_features = []
        for feature_type in self.feature_names.keys():
            col_features.extend(self.feature_names[feature_type])
        return col_features


class LightGBMDataProcessor:
    """Process data for LightGBM models."""

    def __init__(self, config=None):
        """Initialize with configuration."""
        self.config = config or LightGBMDataProcessorConfig()
        self.transformers = {}
        self.features = {"numerical_features": [], "categorical_features": [], "one_hot_features": []}
        self.features_by_source = {}

    def process(
        self,
        candidates: CandidateGeneratorResult,
        article_static_features: ArticleStaticFeatureResult,
        article_dynamic_features: ArticleDynamicFeatureResult,
        customer_static_features: CustomerStaticFeatureResult,
        customer_dynamic_features: CustomerDynamicFeatureResult,
        article_embeddings: ArticleEmbeddingResult,
        sample="train",
    ) -> LightGBMDataResult:
        """
        Process data for LightGBM models.

        Args:
            candidates: Candidates from CandidateGenerator
            article_static_features: Article static features from ArticleStaticFeatureProcessor
            article_dynamic_features: Article dynamic features from ArticleDynamicFeatureProcessor
            customer_static_features: Customer static features from CustomerStaticFeatureProcessor
            customer_dynamic_features: Customer dynamic features from CustomerDynamicFeatureProcessor
            article_embeddings: Article embeddings for similarity calculation
            sample: Sample type ('train', 'valid', 'test')

        Returns:
            LightGBMDataResult containing processed data
        """
        logger.info(f"Processing data for LightGBM: {sample}")

        # Collect feature names
        self._collect_feature_names(
            candidates,
            article_static_features,
            article_dynamic_features,
            customer_static_features,
            customer_dynamic_features,
        )

        # Process train data
        result_train = None
        if sample != "test":
            data_train = self._process_single_dataset(
                candidates,
                article_static_features,
                article_dynamic_features,
                customer_static_features,
                customer_dynamic_features,
                article_embeddings,
                "train",
            )

            # Create result object
            result_train = LightGBMDataResult(
                data=data_train["data"],
                label=data_train["label"],
                group=data_train["group"],
                use_type="train",
                feature_names={
                    "numerical_features": data_train["numerical_features"],
                    "categorical_features": data_train["categorical_features"],
                    "one_hot_features": data_train["one_hot_features"],
                },
                transformers=self.transformers,
                sample="train",
                default_prediction=None,
            )

        # For non-train data, we create another LightGBMDataResult for inference
        result_inference = None
        if sample != "train":
            data_inference = self._process_single_dataset(
                candidates,
                article_static_features,
                article_dynamic_features,
                customer_static_features,
                customer_dynamic_features,
                article_embeddings,
                sample,
            )
            result_inference = LightGBMDataResult(
                data=data_inference["data"],
                label=data_inference["label"],
                group=data_inference["group"],
                use_type="inference",
                feature_names={
                    "numerical_features": data_inference["numerical_features"],
                    "categorical_features": data_inference["categorical_features"],
                    "one_hot_features": data_inference["one_hot_features"],
                },
                transformers=self.transformers,
                sample=sample,
                default_prediction=data_inference.get("default_prediction", None),
            )
        return result_train, result_inference

    def _process_single_dataset(
        self,
        candidates: CandidateGeneratorResult,
        article_static_features: ArticleStaticFeatureResult,
        article_dynamic_features: ArticleDynamicFeatureResult,
        customer_static_features: CustomerStaticFeatureResult,
        customer_dynamic_features: CustomerDynamicFeatureResult,
        article_embeddings: ArticleEmbeddingResult,
        sample="train",
    ) -> Dict:
        """Process a single dataset (train or valid)."""
        logger.info(f"Processing {sample} dataset")
        base = self._prepare_candidates(candidates, sample)

        # Merge with article static features
        if self.config.include_article_static_features and article_static_features is not None:
            logger.debug("Merging with article features")
            base = self._merge_article_static_features(base, article_static_features)

        # Merge with customer static features
        if self.config.include_customer_static_features and customer_static_features is not None:
            logger.debug("Merging with customer features")
            base = self._merge_customer_static_features(base, customer_static_features)

        # Merge with customer dynamic features
        if self.config.include_customer_dynamic_features and customer_dynamic_features is not None:
            logger.debug("Merging with customer dynamic features")
            base = self._merge_customer_dynamic_features(base, customer_dynamic_features, article_embeddings)

        # Merge with article dynamic features
        if self.config.include_article_dynamic_features and article_dynamic_features is not None:
            logger.debug("Merging with article dynamic features")
            base = self._merge_article_dynamic_features(base, article_dynamic_features)

        # Sort dataframe to create groups
        base.sort_values(["customer_id", "week_num", "article_id"], inplace=True)
        base = base.reset_index(drop=True)
        group = base.groupby(["customer_id", "week_num"])["article_id"].count().values
        label = None
        if sample == "train":
            label = base["label"].values

        logger.debug(f"Final base shape: {base.shape}")
        logger.debug(f"Final base columns: {base.columns.tolist()}")

        # Get default predictions if needed
        default_prediction = None
        if sample != "train":
            default_prediction = candidates.default_prediction

        return {
            "data": base,
            "label": label,
            "group": group,
            "numerical_features": self.features["numerical_features"],
            "categorical_features": self.features["categorical_features"],
            "one_hot_features": self.features["one_hot_features"],
            "default_prediction": default_prediction,
        }

    def _prepare_candidates(self, candidates: CandidateGeneratorResult, sample: str) -> List[str]:
        """Prepare candidates for LightGBM data preprocessing."""
        logger.info(f"Preparing candidates for LightGBM data preprocessing: {sample}")
        if sample == "train":
            cols_candidates = candidates.get_feature_list(include_id=True, include_label=True)
            base = candidates.data
        else:
            cols_candidates = candidates.get_feature_list(include_id=True, include_label=False)
            base = candidates.data_inference
        logger.debug(f"Cols candidates: {cols_candidates}")
        base = base[cols_candidates].copy()
        logger.debug(f"Base shape: {base.shape}")
        logger.debug(f"Base columns: {base.columns.tolist()}")
        logger.debug(f"Missing values: {base.isnull().sum()}")
        return base

    def _merge_article_static_features(
        self, base: pd.DataFrame, article_static_features: ArticleStaticFeatureResult
    ) -> pd.DataFrame:
        """Merge base data with article features."""
        logger.debug(f"Base shape before article merge: {base.shape}")

        # Select relevant article features
        article_feature_list = article_static_features.get_feature_list(include_id=True)

        # Merge
        base = base.merge(article_static_features.data[article_feature_list], on="article_id", how="left")

        logger.debug(f"Base shape after article merge: {base.shape}")
        logger.debug(f"Base columns after article merge: {base.columns.tolist()}")
        logger.debug(f"Missing values: {base.isnull().sum()}")
        return base

    def _merge_article_dynamic_features(
        self, base: pd.DataFrame, article_dynamic_features: ArticleDynamicFeatureResult
    ) -> pd.DataFrame:
        """Merge base data with article dynamic features."""
        logger.debug(f"Base shape before article dynamic merge: {base.shape}")

        # Select relevant article dynamic features
        article_dynamic_feature_list = article_dynamic_features.get_feature_list(include_id=True)

        # Merge on article_id and week_num
        # We need to merge using base.week_num = article_dynamic_features.data.week_num - 1 to get previous week's features
        base["prev_week_num"] = base["week_num"] - 1
        base = base.merge(
            article_dynamic_features.data[article_dynamic_feature_list],
            left_on=["article_id", "prev_week_num"],
            right_on=["article_id", "week_num"],
            how="left",
            suffixes=("", "_article_dynamic"),
        )
        base.drop(columns=["prev_week_num", "week_num_article_dynamic"], inplace=True)

        logger.debug(f"Base shape after article dynamic merge: {base.shape}")
        logger.debug(f"Base columns after article dynamic merge: {base.columns.tolist()}")
        logger.debug(f"Missing values: {base.isnull().sum()}")
        return base

    def _merge_customer_static_features(
        self, base: pd.DataFrame, customer_static_features: CustomerStaticFeatureResult
    ) -> pd.DataFrame:
        """Merge base data with customer features."""
        logger.debug(f"Base shape before customer merge: {base.shape}")

        # Select relevant customer features
        customer_feature_list = customer_static_features.get_feature_list(include_id=True)

        # Merge
        base = base.merge(customer_static_features.data[customer_feature_list], on="customer_id", how="left")

        logger.debug(f"Base shape after customer merge: {base.shape}")
        logger.debug(f"Base columns after customer merge: {base.columns.tolist()}")
        logger.debug(f"Missing values: {base.isnull().sum()}")
        return base

    def _merge_customer_dynamic_features(
        self,
        base: pd.DataFrame,
        customer_dynamic_features: CustomerDynamicFeatureResult,
        article_embeddings: ArticleEmbeddingResult,
    ) -> pd.DataFrame:
        """Merge base data with customer dynamic features and calculate embedding similarities.

        Args:
            base: Base DataFrame containing customer_id, article_id, and week_num
            customer_dynamic_features: CustomerDynamicFeatureResult containing dynamic features
            article_embeddings: ArticleEmbeddingResult containing article embeddings

        Returns:
            DataFrame with merged features and similarities
        """
        logger.debug(f"Base shape before customer dynamic merge: {base.shape}")

        # Select relevant customer dynamic features
        customer_dynamic_feature_list = customer_dynamic_features.get_feature_list(include_id=True)

        # Merge on customer_id and week_num
        # We need to merge using base.week_num = customer_dynamic_features.data.week_num - 1 to get previous week's features
        base["prev_week_num"] = base["week_num"] - 1
        base = base.merge(
            customer_dynamic_features.data[customer_dynamic_feature_list],
            left_on=["customer_id", "prev_week_num"],
            right_on=["customer_id", "week_num"],
            how="left",
            suffixes=("", "_customer_dynamic"),
        )
        base.drop(columns=["prev_week_num", "week_num_customer_dynamic"], inplace=True)

        # Calculate cosine similarity between customer and article text embeddings
        logger.debug("Calculating cosine similarities between customer and article embeddings")
        base["text_embedding_similarity"] = calculate_df_batch_cosine_similarity(
            df=base,
            article_embeddings=article_embeddings,
            article_embedding_type="text",
            customer_text_embedding_col="customer_avg_text_embedding",
        )

        logger.debug("Calculating cosine similarities between customer and article image embeddings")
        base["image_embedding_similarity"] = calculate_df_batch_cosine_similarity(
            df=base,
            article_embeddings=article_embeddings,
            article_embedding_type="image",
            customer_text_embedding_col="customer_avg_image_embedding",
        )

        # Drop embedding columns
        base.drop(columns=["customer_avg_text_embedding", "customer_avg_image_embedding"], inplace=True)

        logger.debug(f"Base shape after customer dynamic merge: {base.shape}")
        logger.debug(f"Base columns after customer dynamic merge: {base.columns.tolist()}")
        logger.debug(f"Missing values: {base.isnull().sum()}")
        return base

    def _collect_feature_names(
        self,
        candidates_train: CandidateGeneratorResult,
        article_static_features: ArticleStaticFeatureResult,
        article_dynamic_features: ArticleDynamicFeatureResult,
        customer_static_features: CustomerStaticFeatureResult,
        customer_dynamic_features: CustomerDynamicFeatureResult,
    ) -> None:
        """Collect feature names by type."""
        # Identify feature types
        logger.info("Collecting feature names from datasets")
        datasets_dict = {
            "candidates": candidates_train,
        }
        if self.config.include_article_static_features:
            datasets_dict["articles_static"] = article_static_features
        if self.config.include_article_dynamic_features:
            datasets_dict["articles_dynamic"] = article_dynamic_features
        if self.config.include_customer_static_features:
            datasets_dict["customers_static"] = customer_static_features
        if self.config.include_customer_dynamic_features:
            datasets_dict["customers_dynamic"] = customer_dynamic_features

        for key, dataset in datasets_dict.items():
            logger.debug(f"Processing features from {key} dataset")
            self.features_by_source[key] = {}
            for feature_type in ["categorical_features", "numerical_features", "one_hot_features"]:
                self.features_by_source[key][feature_type] = dataset.feature_names.get(feature_type, [])
                self.features[feature_type].extend(dataset.feature_names.get(feature_type, []))

        # Add additional new features generated by the pipeline
        # New features added by customers dynamic features
        self.features["numerical_features"].extend(["text_embedding_similarity", "image_embedding_similarity"])
        self.features_by_source["customers_dynamic"]["numerical_features"].extend(
            ["text_embedding_similarity", "image_embedding_similarity"]
        )

        logger.debug(f"Collected features: {self.features}")
        logger.debug(f"Features by source: {self.features_by_source}")


class LightGBMDataPipeline:
    """End-to-end pipeline for LightGBM data preparation."""

    def __init__(self, config=None):
        """Initialize pipeline with config."""
        self.config = config or LightGBMDataProcessorConfig()
        self.processor = None

    def setup(self, config=None):
        """Set up the pipeline with configuration."""
        logger.info("Setting up LightGBMDataPipeline")
        if config is not None:
            self.config = config

        logger.info("Setting up LightGBMDataPipeline")
        logger.debug(f"Using configuration: {json.dumps(self.config.to_dict(), indent=2)}")
        self.processor = LightGBMDataProcessor(self.config)
        return self

    def _save_result(self, result: LightGBMDataResult):
        """Save result to disk."""
        logger.info(f"Saving result to disk for LightGBMDataResult of type {result.use_type}")
        output_path = get_path_to_lightgbm_data(
            self.config.sample, result.use_type, self.config.subsample, self.config.seed
        )
        logger.debug(f"Saving result to disk: {output_path}")
        result.save(output_path)

    def run(self):
        if self.processor is None:
            raise ValueError("Pipeline is not set up. Call setup() before running the pipeline.")

        # Load data
        (
            candidates_result,
            article_static_features_result,
            article_dynamic_features_result,
            customer_static_features_result,
            customer_dynamic_features_result,
            article_embeddings_result,
        ) = self._load_data()

        # Process data
        result_train, result_inference = self.processor.process(
            candidates_result,
            article_static_features_result,
            article_dynamic_features_result,
            customer_static_features_result,
            customer_dynamic_features_result,
            article_embeddings_result,
            self.config.sample,
        )

        # Save results
        if result_train is not None:
            self._save_result(result_train)
        if result_inference is not None:
            self._save_result(result_inference)

        return result_train, result_inference

    def _load_data(self):
        """Load data if a path is provided and loading is enabled."""
        if self.config.use_default_data_paths:
            logger.info("Using default data paths")
            candidates_path = get_path_to_candidates(self.config.sample, self.config.subsample, self.config.seed)
            articles_static_features_path = get_path_to_article_features(
                feature_type="static", subsample=self.config.subsample, seed=self.config.seed
            )
            articles_dynamic_features_path = get_path_to_article_features(
                feature_type="dynamic", subsample=self.config.subsample, seed=self.config.seed
            )
            customers_static_features_path = get_path_to_customers_features(
                feature_type="static", subsample=self.config.subsample, seed=self.config.seed
            )
            customers_dynamic_features_path = get_path_to_customers_features(
                feature_type="dynamic", subsample=self.config.subsample, seed=self.config.seed
            )
            article_embeddings_path = get_path_to_article_features(
                feature_type="embeddings", subsample=self.config.subsample, seed=self.config.seed
            )

        else:
            raise NotImplementedError("Custom data loading is not implemented yet.")

        # Load data results
        logger.info("Loading data from disk")
        candidates_result = CandidateGeneratorResult.load(candidates_path)
        articles_static_features_result = ArticleStaticFeatureResult.load(articles_static_features_path)
        articles_dynamic_features_result = ArticleDynamicFeatureResult.load(articles_dynamic_features_path)
        customers_static_features_result = CustomerStaticFeatureResult.load(customers_static_features_path)
        customers_dynamic_features_result = CustomerDynamicFeatureResult.load(customers_dynamic_features_path)
        article_embeddings_result = ArticleEmbeddingResult.load(article_embeddings_path)
        logger.info("Data loaded successfully")

        return (
            candidates_result,
            articles_static_features_result,
            articles_dynamic_features_result,
            customers_static_features_result,
            customers_dynamic_features_result,
            article_embeddings_result,
        )
