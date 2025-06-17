import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.config import DEFAULT_CANDIDATE_GENERATION_CONFIG
from src.feature_extraction import (
    MAX_WEEK_DATE,
    MAX_WEEK_NUM,
    load_optimized_raw_data,
)
from src.sampling.manager import NegativeSamplingManager

logger = logging.getLogger(__name__)


# ======================================================================================================================
# Utility functions
# ======================================================================================================================
def get_transactions_by_period(transactions: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp):
    """Filter transactions by a given date range."""
    logger.info(f"Filtering transactions from {start_date} to {end_date}")
    return transactions[(transactions["t_dat"] >= start_date) & (transactions["t_dat"] <= end_date)].copy()


def get_week_starting_wed(s: Union[pd.Series, pd.Timestamp]) -> pd.Series:
    if isinstance(s, pd.Series):
        return s.dt.to_period("W-TUE").dt.start_time
    elif isinstance(s, pd.Timestamp):
        return s.to_period("W-TUE").start_time
    else:
        raise ValueError("Input must be a pandas Series or Timestamp.")


def week_from_week_num(s: pd.Series) -> pd.Series:
    """Convert week number to week start date."""
    return MAX_WEEK_DATE + pd.to_timedelta(1, unit="D") - pd.to_timedelta(104 - s + 1, unit="W")


def week_num_from_week(s: pd.Series) -> pd.Series:
    """Convert week start date to week number."""
    return MAX_WEEK_NUM - (MAX_WEEK_DATE - s).days // 7  # 104 is the maximum week number in the dataset


def get_path_to_candidates(sample: str, subsample: float, seed: int) -> str:
    """Get path to candidates."""
    if subsample < 1:
        return f"../data/preprocessed/candidate_generator/{sample}/subsample_{subsample}_{seed}"
    else:
        return f"../data/preprocessed/candidate_generator/{sample}/full"


# ======================================================================================================================
# CandidateGenerator class
# ======================================================================================================================
@dataclass
class CandidateGeneratorResult:
    """Results from candidate generation process"""

    data: pd.DataFrame  # Main data frame with candidates
    data_inference: pd.DataFrame = None  # Data frame with candidates for inference
    label: Optional[np.ndarray] = None  # Target labels if available
    feature_names: Dict[str, List[str]] = None  # Feature metadata
    sample: str = "train"  # Sample type (train, valid, test)
    default_prediction: Optional[np.ndarray] = None  # Default predictions for inference

    def __post_init__(self):
        """Validate the result object after initialization"""
        if self.feature_names is None:
            self.feature_names = {}

    def save(self, path_to_dir: str):
        """Save result to disk using component-wise saving for efficiency."""
        logger.info(f"Saving CandidateGeneratorResult to {path_to_dir}")
        # Create directory if it doesn't exist already
        os.makedirs(path_to_dir, exist_ok=True)

        # Save components separately for efficiency
        self.data.to_parquet(f"{path_to_dir}/data.parquet")
        logger.info(f"CandidateGeneratorResult data saved to {path_to_dir}/data.parquet")

        if self.data_inference is not None:
            self.data_inference.to_parquet(f"{path_to_dir}/data_inference.parquet")
            logger.info(f"CandidateGeneratorResult data_inference saved to {path_to_dir}/data_inference.parquet")

        if self.label is not None:
            np.save(f"{path_to_dir}/label.npy", self.label)
            logger.info(f"CandidateGeneratorResult label saved to {path_to_dir}/label.npy")

        with open(f"{path_to_dir}/feature_names.json", "w") as f:
            json.dump(self.feature_names, f)
            logger.info(f"CandidateGeneratorResult feature_names saved to {path_to_dir}/feature_names.json")

        if self.default_prediction is not None:
            np.save(f"{path_to_dir}/default_prediction.npy", self.default_prediction)
            logger.info(f"CandidateGeneratorResult default_prediction saved to {path_to_dir}/default_prediction.npy")

        # Save metadata about the saved components
        metadata = {
            "has_label": self.label is not None,
            "columns": list(self.data.columns),
            "feature_name_keys": list(self.feature_names.keys() if self.feature_names else []),
            "has_data_inference": self.data_inference is not None,
            "has_default_predictions": self.default_prediction is not None,
            "sample": self.sample,
        }
        with open(f"{path_to_dir}/metadata.json", "w") as f:
            json.dump(metadata, f)
            logger.info(f"CandidateGeneratorResult metadata saved to {path_to_dir}/metadata.json")

    @classmethod
    def load(cls, path_to_dir: str):
        """Load result from disk."""
        logger.info(f"Loading CandidateGeneratorResult from {path_to_dir}")
        # Load metadata to check what's available
        with open(f"{path_to_dir}/metadata.json", "r") as f:
            metadata = json.load(f)
        logger.info(f"Metadata loaded: {metadata}")

        # Load components
        data = pd.read_parquet(f"{path_to_dir}/data.parquet")
        logger.info(f"CandidateGeneratorResult data loaded from {path_to_dir}/data.parquet")

        data_inference = None
        if metadata["has_data_inference"]:
            data_inference = pd.read_parquet(f"{path_to_dir}/data_inference.parquet")
            logger.info(f"CandidateGeneratorResult data_inference loaded from {path_to_dir}/data_inference.parquet")

        label = None
        if metadata["has_label"]:
            label = np.load(f"{path_to_dir}/label.npy")
            logger.info(f"CandidateGeneratorResult label loaded from {path_to_dir}/label.npy")

        sample = metadata["sample"]
        default_prediction = None
        if metadata["has_default_predictions"]:
            default_prediction = np.load(f"{path_to_dir}/default_prediction.npy")
            logger.info(f"CandidateGeneratorResult default_prediction loaded from {path_to_dir}/default_prediction.npy")

        with open(f"{path_to_dir}/feature_names.json", "r") as f:
            feature_names = json.load(f)
        logger.info(f"CandidateGeneratorResult feature_names loaded from {path_to_dir}/feature_names.json")

        return cls(
            data=data,
            data_inference=data_inference,
            label=label,
            feature_names=feature_names,
            sample=sample,
            default_prediction=default_prediction,
        )

    def get_feature_list(self, include_id: bool = True, include_label: bool = True) -> List[str]:
        """Get a flattened list of all features."""
        features = []
        for feature_type, feature_list in self.feature_names.items():
            if not include_id and feature_type in ["id_columns", "metadata_columns"]:
                continue
            if not include_label and feature_type == "target_columns":
                continue
            features.extend(feature_list)

        return features


@dataclass
class CandidateGeneratorPipelineConfig:
    """Configuration for candidate generation pipeline"""

    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    history_start_date: pd.Timestamp
    n_sample_week_threshold: int = -1
    negative_sample_strategies: Dict[str, Dict[str, Any]] = None
    inference_sample_strategies: Dict[str, Dict[str, Any]] = None
    subsample: float = 1.0  # Fraction of data to use for training
    seed: int = 42  # Random seed for reproducibility
    restrict_positive_samples: bool = (
        False  # If True, only keep positive samples that are in the candidate generation sources
    )
    neg_to_pos_ratio: float = 30.0  # Expected ratio of negative to positive samples. If -1, no adjustment is done.

    def __post_init__(self):
        """Set default values if not provided"""
        if self.negative_sample_strategies is None:
            self.negative_sample_strategies = {
                "popularity": {"top_k_items": 12},
                "repurchase": {"strategy": "last_k_items", "k": 12},
            }
        if self.inference_sample_strategies is None:
            self.inference_sample_strategies = {
                "popularity": {"top_k_items": 12},
                "repurchase": {"strategy": "last_k_items", "k": 12},
            }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary"""
        # Convert string dates to timestamps
        if isinstance(config_dict.get("train_start_date"), str):
            config_dict["train_start_date"] = pd.to_datetime(config_dict["train_start_date"])
        if isinstance(config_dict.get("train_end_date"), str):
            config_dict["train_end_date"] = pd.to_datetime(config_dict["train_end_date"])
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary with JSON-serializable values."""
        result = asdict(self)

        # Convert timestamps to strings
        for key, value in result.items():
            if isinstance(value, pd.Timestamp):
                result[key] = value.isoformat()
            # Handle nested dictionaries
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, pd.Timestamp):
                        value[k] = v.isoformat()

        return result

    @classmethod
    def create_default(cls):
        """Create a default configuration"""
        return cls.from_dict(DEFAULT_CANDIDATE_GENERATION_CONFIG)


class CandidateGenerator:
    """Class to generate candidates for training and inference."""

    def __init__(self, config: dict):
        """Initialize the transaction processor with optional logger."""
        self.train_start_date = config["train_start_date"]
        self.train_end_date = config["train_end_date"]
        self.train_start_week_num = week_num_from_week(self.train_start_date)
        self.train_end_week_num = week_num_from_week(self.train_end_date)
        if self.train_start_date - config["history_start_date"] < pd.Timedelta(days=7):
            raise ValueError(
                f"History start date {config['history_start_date']} is less than 7 days before train start date {self.train_start_date}"
            )
        self.history_start_date = config["history_start_date"]

        self.n_sample_week_threshold = config["n_sample_week_threshold"]
        self.negative_sample_strategies = config["negative_sample_strategies"]
        self.inference_sample_strategies = config["inference_sample_strategies"]
        self.restrict_positive_samples = config["restrict_positive_samples"]
        self.neg_to_pos_ratio = config["neg_to_pos_ratio"]

    @staticmethod
    def prepare_base_transactions_data(
        raw_transactions: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        customers_ids: list[str] = None,
        articles_ids: list[int] = None,
    ):
        logger.info("Preparing base transactions data")
        logger.debug(f"Shape of transactions before filtering: {raw_transactions.shape}")
        transactions = raw_transactions.copy()
        if "week" not in transactions.columns:
            logger.debug("Adding week column to transactions")
            transactions["week"] = get_week_starting_wed(transactions["t_dat"])
        if customers_ids is not None:
            logger.debug("Filtering transactions by customer_ids")
            transactions = transactions[transactions["customer_id"].isin(customers_ids)].copy()
            logger.debug(f"Number of transactions after filtering by customer_ids: {len(transactions)}")
        if articles_ids is not None:
            logger.debug("Filtering transactions by article_ids")
            transactions = transactions[transactions["article_id"].isin(articles_ids)].copy()
            logger.debug(f"Number of transactions after filtering by article_ids: {len(transactions)}")
        transactions = get_transactions_by_period(transactions, start_date, end_date)
        logger.debug(f"Shape of transactions after filtering: {transactions.shape}")
        return transactions

    @property
    def numerical_features(self):
        # Numerical features
        return [
            "price",
            "week_num",
            "bestseller_rank",
        ]

    @property
    def categorical_features(self):
        return [
            # "sales_channel_id",
            "month",
        ]

    @property
    def one_hot_features(self):
        return []

    @property
    def id_columns(self):
        """ID columns that should be preserved but not used as features."""
        return [
            "customer_id",
            "article_id",
        ]

    @property
    def metadata_columns(self):
        """Metadata columns that should be preserved but not used as features."""
        return [
            "week",
            "source",
        ]

    @property
    def target_columns(self):
        """Target or label columns."""
        return ["label"]

    def _prepare_positive_samples(
        self, transactions: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp, n_sample_week_threshold: int
    ):
        """Prepare positive samples between start_date and end_date. Keep only up to n_samples_week_threshold for each
        customer each week."""
        logger.info("Preparing positive samples")
        transactions1 = get_transactions_by_period(transactions, start_date, end_date)

        # For customer with more than n_sample_week_threshold purchases in a week, randomly select n_sample_week_threshold
        # to keep
        if n_sample_week_threshold > 0:
            # Permute the rows and take the top n_sample_week_threshold for each customer and week
            transactions1 = transactions1.sample(frac=1).reset_index(drop=True)
            transactions1.sort_values(["customer_id", "week_num"], inplace=True)
            transactions1 = (
                transactions1.groupby(["customer_id", "week_num"]).head(n_sample_week_threshold).reset_index(drop=True)
            )

        transactions1.drop(columns=["t_dat", "week"], inplace=True)

        transactions1.drop_duplicates(subset=["customer_id", "week_num", "article_id"], inplace=True)
        logger.debug(f"Number of positive samples: {len(transactions1)}")

        transactions1["source"] = "positive"

        return transactions1

    def _generate_negative_samples(
        self,
        # Transactions should contain the full history (do not have to include the actual labels)
        # If sample == "train", it is used to remove negative samples that are in the transactions dataframe (same
        # customer_id, week_num, article_id)
        transactions: pd.DataFrame,
        # Customer_id, week_num pairs that are unique and observed in actual data
        unique_transactions: pd.DataFrame,
        # Week of the first week of the training data
        train_start_week_num: int,
        # Week of the last week of the training data
        train_end_week_num: int,
        sample: str,
        customers_ids=None,
        restrict_positive_samples: bool = False,
    ):
        """Generate negative samples using configured strategies.

        Uses the NegativeSamplingManager to orchestrate the generation of negative samples
        from different strategies and combine them.

        Args:
            transactions: DataFrame with transaction history
            unique_transactions: DataFrame with unique customer-week pairs
            train_start_week_num: Start week number for training
            train_end_week_num: End week number for training
            sample: "train" or inference sample type
            customers_ids: Optional list of customer IDs to filter

        Returns:
            Tuple of (negative_samples, default_prediction, popular_items)
        """
        logger.info(f"Generating negative samples for {sample}")

        # Determine which sampling strategies to use based on sample type
        if sample == "train":
            sampling_strategies = self.negative_sample_strategies
        else:
            sampling_strategies = self.inference_sample_strategies

        # Create sampling manager
        sampling_manager = NegativeSamplingManager(sampling_strategies=sampling_strategies)

        # Generate samples
        restrict_negative_samples = not restrict_positive_samples
        return sampling_manager.generate_samples(
            transactions=transactions,
            unique_customer_week_pairs=unique_transactions,
            week_num_start=train_start_week_num,
            week_num_end=train_end_week_num,
            sample_type=sample,
            customers_ids=customers_ids,
            restrict_negative_samples=restrict_negative_samples,
        )

    def _combine_positive_negative_samples(self, positive_samples: pd.DataFrame, negative_samples: pd.DataFrame):
        """Combine positive and negative samples into a single dataframe and create labels."""
        # Combine positive and negative samples
        logger.info("Combining positive and negative samples")
        logger.debug(f"Number of positive samples: {len(positive_samples)}")
        logger.debug(f"Number of negative samples: {len(negative_samples)}")

        if self.restrict_positive_samples:
            # Only keep positive samples that are in the candidate generation sources
            combined_samples = negative_samples.merge(
                positive_samples[["customer_id", "week_num", "article_id"]],
                on=["customer_id", "week_num", "article_id"],
                how="left",
                indicator=True,
            )
            combined_samples["label"] = combined_samples["_merge"].apply(lambda x: 1 if x == "both" else 0)
            combined_samples.drop(columns=["_merge"], inplace=True)
        else:
            positive_samples["label"] = 1
            negative_samples["label"] = 0
            combined_samples = pd.concat([positive_samples, negative_samples], axis=0, ignore_index=True)
        logger.debug(f"Number of combined samples: {len(combined_samples)}")
        return combined_samples

    def _adjust_neg_to_pos_ratio(self, combined_samples: pd.DataFrame):
        """Adjust the negative to positive ratio to the expected ratio."""
        logger.info(f"Adjusting negative to positive ratio to {self.neg_to_pos_ratio}")
        # Count the number of positive and negative samples
        num_positive = combined_samples["label"].sum()
        num_negative = len(combined_samples) - num_positive
        current_ratio = num_negative / num_positive
        if current_ratio > self.neg_to_pos_ratio:
            logger.info(
                f"Current ratio {current_ratio} is greater than expected ratio {self.neg_to_pos_ratio}, subsampling negative samples"
            )
            # Subsample negative samples to the expected ratio
            subsample_ratio = self.neg_to_pos_ratio / current_ratio
            positive_samples = combined_samples[combined_samples["label"] == 1]
            negative_samples = combined_samples[combined_samples["label"] == 0]
            negative_samples = negative_samples.groupby(["customer_id", "week_num"]).sample(frac=subsample_ratio)
            combined_samples = pd.concat([positive_samples, negative_samples], axis=0, ignore_index=True)

            observed_ratio = len(negative_samples) / len(positive_samples)
            logger.info("Finished subsampling negative samples")
            logger.info(f"Observed negative to positive ratio {observed_ratio}")
            logger.info(f"Total number of samples {len(combined_samples)}")

        return combined_samples

    def _add_date_features(self, transactions: pd.DataFrame, col_date: str) -> pd.DataFrame:
        """Add date features to the transactions dataframe."""
        logger.info("Adding date features")
        transactions["year"] = transactions[col_date].dt.year
        transactions["month"] = transactions[col_date].dt.month
        transactions["week"] = transactions["week"].dt.strftime("%Y-%m-%d")

        return transactions

    def _add_best_seller_rank(self, transactions: pd.DataFrame, top_k_articles_by_week: pd.DataFrame) -> pd.DataFrame:
        """Add best seller rank to the transactions dataframe."""
        logger.info("Adding best seller rank")
        transactions["prev_week_num"] = transactions["week_num"] - 1
        transactions = transactions.merge(
            top_k_articles_by_week[["week_num", "article_id", "bestseller_rank"]],
            left_on=["prev_week_num", "article_id"],
            right_on=["week_num", "article_id"],
            how="left",
            suffixes=("", "_bestseller"),
        )
        transactions.fillna({"bestseller_rank": 999}, inplace=True)
        transactions.drop(columns=["prev_week_num", "week_num_bestseller"], inplace=True)
        return transactions

    def _generate_transaction_features(
        self, transactions: pd.DataFrame, top_k_articles_by_week: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate features for the transactions dataframe."""
        logger.info("Generating transaction features")
        # Generate week from week number
        transactions["week"] = week_from_week_num(transactions["week_num"])
        transactions = self._add_date_features(transactions, "week")
        transactions = self._add_best_seller_rank(transactions, top_k_articles_by_week)
        return transactions

    def _prepare_for_training(
        self,
        raw_transactions: pd.DataFrame,
        articles: pd.DataFrame,
        train_start_date: pd.Timestamp,
        train_end_date: pd.Timestamp,
        train_start_week_num: int,
        train_end_week_num: int,
    ) -> pd.DataFrame:
        """Prepare transactions data from raw data for training."""

        # Select the transactions between history_start_date and train_end_date for the selected customer_ids and articles_ids
        transactions = self.prepare_base_transactions_data(
            raw_transactions,
            self.history_start_date,  # This is one week before the training dat in config.
            train_end_date,
        )

        # Prepare positive samples
        positive_transactions = self._prepare_positive_samples(
            transactions, train_start_date, train_end_date, self.n_sample_week_threshold
        )
        customers_ids = positive_transactions["customer_id"].unique()
        unique_transactions = positive_transactions[["customer_id", "week_num"]].drop_duplicates()

        negative_transactions, default_prediction, top_k_articles_by_week = self._generate_negative_samples(
            transactions=transactions,
            unique_transactions=unique_transactions,
            train_start_week_num=train_start_week_num,
            train_end_week_num=train_end_week_num,
            restrict_positive_samples=self.restrict_positive_samples,
            sample="train",
            customers_ids=customers_ids,
        )

        # Combine positive and negative samples
        combined_samples = self._combine_positive_negative_samples(positive_transactions, negative_transactions)

        # Adjust negative to positive ratio
        if self.neg_to_pos_ratio != -1:
            combined_samples = self._adjust_neg_to_pos_ratio(combined_samples)

        # Generate features
        combined_samples = self._generate_transaction_features(combined_samples, top_k_articles_by_week)

        return CandidateGeneratorResult(
            data=combined_samples,
            label=combined_samples["label"].to_numpy(),
            feature_names={
                "categorical_features": self.categorical_features,
                "numerical_features": self.numerical_features,
                "one_hot_features": self.one_hot_features,
                "id_columns": self.id_columns,
                "metadata_columns": self.metadata_columns,
                "target_columns": self.target_columns,
            },
            sample="train",
            default_prediction=default_prediction,
        )

    def _get_unique_customer_week_pair(self, customers_ids, inference_week, inference_start_week_num):
        """Get unique customer-week pairs from the transactions dataframe."""
        logger.info("Getting unique customer-week pairs for inference")
        # Create a DataFrame with unique customer-week pairs
        unique_pairs = pd.DataFrame(
            {
                "customer_id": customers_ids,
                "week": inference_week,
                "week_num": inference_start_week_num,
            }
        )
        logger.debug(f"Shape of unique_pairs: {unique_pairs.shape}")
        return unique_pairs

    def _prepare_for_inference(
        self,
        # raw_transactions should be the full training transactions data
        raw_transactions: pd.DataFrame,
        transactions_for_inference: pd.DataFrame,
        articles: pd.DataFrame,
        sample: str,
    ):
        # The method for preparing for inference data is different from training because we only consider the inference
        # week's consumers but leveraging historical transactions data for generating candidates
        logger.info("Preparing transactions data for inference")

        # Assume that inference data is only for one week
        inference_start_date = transactions_for_inference.t_dat.min()
        inference_end_date = transactions_for_inference.t_dat.max()
        inference_week_num = transactions_for_inference.week_num.min()
        inference_week = get_week_starting_wed(inference_start_date)

        # Prepare a "training" data that contains positive samples for the inference week
        # Used for early stopping purpose
        results = self._prepare_for_training(
            raw_transactions=pd.concat([raw_transactions, transactions_for_inference], ignore_index=True),
            articles=articles,
            train_start_date=inference_start_date,
            train_end_date=inference_end_date,
            train_start_week_num=inference_week_num,
            train_end_week_num=inference_week_num,
        )

        # Generate only negative samples ignoring the positive samples
        customers_ids = transactions_for_inference.customer_id.unique()
        unique_pairs = self._get_unique_customer_week_pair(customers_ids, inference_week, inference_week_num)

        # Clean up raw transactions for candidate recommendation
        # Do not include any filter on customer_id
        past_transactions = self.prepare_base_transactions_data(
            raw_transactions,
            self.history_start_date,  # This is one week before the training date.
            inference_start_date - pd.DateOffset(days=1),
            None,
            None,
        )

        candidates, default_prediction, top_k_articles_by_week = self._generate_negative_samples(
            transactions=past_transactions,
            unique_transactions=unique_pairs,
            train_start_week_num=inference_week_num,
            train_end_week_num=inference_week_num,
            sample=sample,
            customers_ids=customers_ids,
            restrict_positive_samples=self.restrict_positive_samples,
        )

        # Generate features
        candidates = self._generate_transaction_features(candidates, top_k_articles_by_week)

        # Update CandidateGeneratorResult with inference data
        results.sample = sample
        results.data_inference = candidates
        results.default_prediction = default_prediction

        # Return
        return results

    def prepare_data(
        self,
        raw_transactions: pd.DataFrame,
        articles: pd.DataFrame,
        sample: str,
        transactions_for_inference: pd.DataFrame = None,
    ) -> dict:
        if sample == "train":
            return self._prepare_for_training(
                raw_transactions=raw_transactions,
                articles=articles,
                train_start_date=self.train_start_date,
                train_end_date=self.train_end_date,
                train_start_week_num=self.train_start_week_num,
                train_end_week_num=self.train_end_week_num,
            )
        elif sample in ["valid", "test"]:
            return self._prepare_for_inference(
                raw_transactions,
                transactions_for_inference,
                articles,
                sample,
            )
        else:
            raise ValueError("sample must be one of ['train', 'valid', 'test']")


class CandidateGeneratorPipeline:
    """End-to-end pipeline for candidate generation"""

    def __init__(self, config: Optional[CandidateGeneratorPipelineConfig] = None):
        """Initialize pipeline with config"""
        self.config = config
        self.generator = None

    def setup(self, config: CandidateGeneratorPipelineConfig = None):
        """Set up the pipeline with a config"""
        logger.info("Setting up CandidateGeneratorPipeline")
        if config is not None:
            self.config = config
        if self.config is None:
            raise ValueError("Config must be provided")
        logger.debug(f"Pipeline config: {json.dumps(self.config.to_dict(), indent=2)}")
        self.generator = CandidateGenerator(asdict(self.config))
        return self

    def _load_data(self):
        """Load data for training and inference"""
        logger.info("Loading data for CandidateGeneratorPipeline")
        transactions_train = load_optimized_raw_data(
            data_type="transactions", sample="train", subsample=self.config.subsample, seed=self.config.seed
        )
        transactions_valid = load_optimized_raw_data(
            data_type="transactions", sample="valid", subsample=self.config.subsample, seed=self.config.seed
        )
        transactions_test = load_optimized_raw_data(
            data_type="transactions", sample="test", subsample=self.config.subsample, seed=self.config.seed
        )
        articles = load_optimized_raw_data(data_type="articles", subsample=self.config.subsample, seed=self.config.seed)
        return transactions_train, transactions_valid, transactions_test, articles

    def run_train(self, transactions: pd.DataFrame, articles: pd.DataFrame) -> CandidateGeneratorResult:
        """Run pipeline for training data"""
        logger.info("Running pipeline for training data")
        if self.generator is None:
            raise ValueError("Pipeline not set up. Call setup() first.")

        # Generate candidates
        result = self.generator.prepare_data(transactions, articles, sample="train")

        # Save results
        path_to_dir = "../data/preprocessed/candidate_generator/train"
        if self.config.subsample < 1.0:
            path_to_dir += f"/subsample_{self.config.subsample}_{self.config.seed}"
        else:
            path_to_dir += "/full"
        result.save(path_to_dir)

        return result

    def run_inference(
        self,
        transactions_history: pd.DataFrame,
        transactions_inference: pd.DataFrame,
        articles: pd.DataFrame,
        sample: str = "valid",
    ) -> CandidateGeneratorResult:
        """Run pipeline for inference"""
        if self.generator is None:
            raise ValueError("Pipeline not set up. Call setup() first.")

        # Generate candidates
        result = self.generator.prepare_data(
            transactions_history, articles, sample=sample, transactions_for_inference=transactions_inference
        )

        # Save results
        path_to_dir = get_path_to_candidates(sample, self.config.subsample, self.config.seed)
        result.save(path_to_dir)

        return result

    def run(self):
        transactions_train, transactions_valid, transactions_test, articles = self._load_data()
        result_train = self.run_train(transactions_train, articles)
        result_valid = self.run_inference(
            transactions_history=transactions_train,
            transactions_inference=transactions_valid,
            articles=articles,
            sample="valid",
        )
        result_test = self.run_inference(
            transactions_history=pd.concat([transactions_train, transactions_valid], axis=0, ignore_index=True),
            transactions_inference=transactions_test,
            articles=articles,
            sample="test",
        )
        return result_train, result_valid, result_test
