import json
import logging

import pandas as pd

from src.utils.core_utils import human_readable_size

logger = logging.getLogger(__name__)

PATH_TRANSACTIONS_RAW_DATA = "../data/raw/transactions_train.csv"
PATH_ARTICLES_RAW_DATA = "../data/raw/articles.csv"
PATH_CUSTOMERS_RAW_DATA = "../data/raw/customers.csv"

VAL_CATCHALL = "unknown"
MIN_WEEK = pd.to_datetime("2018-09-20")

# Maximum number of weeks in the dataset
MAX_WEEK_NUM = 104

# Maximum date in the raw transactions data
MAX_WEEK_DATE = pd.to_datetime("2020-09-22")

TEST_START_DATE = pd.to_datetime("2020-09-16")
VALID_START_DATE = pd.to_datetime("2020-09-09")
WEEK_NUM_VALID = 103
WEEK_NUM_TEST = 104


# ======================================================================================================================
# Utility functions
# ======================================================================================================================
def customer_hex_id_to_int(series):
    return series.str[-16:].apply(hex_id_to_int)


def hex_id_to_int(str):
    return int(str[-16:], 16)


def article_id_str_to_int(series):
    return series.astype("int32")


def article_id_int_to_str(series):
    return "0" + series.astype("str")


def load_optimized_raw_data(
    data_type: str, sample: str = "train", subsample: float = 1, seed: int = 42
) -> pd.DataFrame:
    """
    Load optimized raw data from the H&M dataset.

    Args:
        data_type: Type of data ('transactions', 'customers', or 'articles', 'candidates_to_articles_mapping')
        sample: Data sample type ('train', 'valid', or 'test'). Only used for transactions data.
        subsample: Subsampling fraction (0-1)
        seed: Random seed for reproducibility

    Returns:
        DataFrame containing the requested data
    """
    logger.info(f"Loading optimized raw data from {data_type} {sample} {subsample} {seed}")
    path = RawDataOptimizer.get_output_paths(data_type, sample=sample, subsample=subsample, seed=seed)
    logger.debug(f"Loading data from {path}")
    if data_type in ["transactions", "customers", "articles"]:
        return pd.read_parquet(path)
    elif data_type == "candidates_to_articles_mapping":
        with open(path, "r") as f:
            # Convert string keys to int
            mapping = json.load(f)
            return {int(k): v for k, v in mapping.items()}
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def get_path_to_raw_data(data_type: str, sample: str = "train", subsample: float = 1, seed: int = 42) -> str:
    """
    Get output paths for data files.

    Args:
        data_type: Type of data ('transactions', 'customers', or 'articles')
        sample: Data sample type ('train', 'valid', or 'test'). Only used for transactions data.
        subsample: Subsampling fraction (0-1)
        seed: Random seed for reproducibility

    Returns:
        Path to the requested data file
    """
    base_path = "../data"

    if subsample < 1:
        subsample_str = f"_sample_{subsample}_{seed}"
    else:
        subsample_str = ""

    if data_type == "transactions":
        if sample == "train":
            return f"{base_path}/transactions_train{subsample_str}.parquet"
        elif sample == "valid":
            return f"{base_path}/transactions_valid{subsample_str}.parquet"
        elif sample == "test":
            return f"{base_path}/transactions_test{subsample_str}.parquet"
        else:
            raise ValueError("Sample must be one of ['train', 'valid', 'test']")
    elif data_type == "customers":
        return f"{base_path}/customers{subsample_str}.parquet"
    elif data_type == "articles":
        return f"{base_path}/articles{subsample_str}.parquet"
    elif data_type == "candidates_to_articles_mapping":
        if sample == "valid":
            return f"{base_path}/candidates_to_articles_mapping_valid{subsample_str}.json"
        elif sample == "test":
            return f"{base_path}/candidates_to_articles_mapping_test{subsample_str}.json"
        else:
            raise ValueError("Sample must be one of ['valid', 'test']")
    else:
        raise ValueError("Data type must be one of ['transactions', 'customers', 'articles']")


# ======================================================================================================================
# Raw data optimizer
# ======================================================================================================================
class RawDataOptimizer:
    """Load and optimize raw data from the H&M dataset."""

    def __init__(self, subsample: int = 1, seed: int = 42):
        self.subsample = subsample
        self.seed = seed

    @staticmethod
    def get_output_paths(data_type: str, sample: str = "train", subsample: float = 1, seed: int = 42) -> str:
        """
        Get output paths for data files.

        Args:
            data_type: Type of data ('transactions', 'customers', or 'articles')
            sample: Data sample type ('train', 'valid', or 'test'). Only used for transactions data.
            subsample: Subsampling fraction (0-1)
            seed: Random seed for reproducibility

        Returns:
            Path to the requested data file
        """
        return get_path_to_raw_data(data_type, sample, subsample, seed)

    def _read_raw_data(self) -> pd.DataFrame:
        """Read raw transactions data from the CSV file."""
        logger.info("Reading raw transactions data")
        transactions = pd.read_csv(PATH_TRANSACTIONS_RAW_DATA, dtype={"article_id": "str"})
        customers = pd.read_csv(PATH_CUSTOMERS_RAW_DATA)
        articles = pd.read_csv(PATH_ARTICLES_RAW_DATA, dtype={"article_id": "str"})
        transactions_size = human_readable_size(transactions.memory_usage(deep=True).sum())
        customers_size = human_readable_size(customers.memory_usage(deep=True).sum())
        articles_size = human_readable_size(articles.memory_usage(deep=True).sum())
        logger.debug(f"Shape of raw transactions data: {transactions.shape}")
        logger.debug(f"Shape of raw customers data: {customers.shape}")
        logger.debug(f"Shape of raw articles data: {articles.shape}")
        logger.debug(f"Size of raw transactions data: {transactions_size}")
        logger.debug(f"Size of raw customers data: {customers_size}")
        logger.debug(f"Size of raw articles data: {articles_size}")
        return transactions, customers, articles

    def _clean_transactions_data(self, transactions: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning transactions data")
        # Clean customer ID from string to int
        transactions["customer_id"] = customer_hex_id_to_int(transactions["customer_id"])

        # Clean article ID
        transactions["article_id"] = article_id_str_to_int(transactions["article_id"])

        # Clean dates
        transactions.t_dat = pd.to_datetime(transactions.t_dat, format="%Y-%m-%d")
        transactions["week_num"] = 104 - (transactions.t_dat.max() - transactions.t_dat).dt.days // 7

        # Clean dtypes
        dtypes_dict = {
            "week_num": "int8",
            "sales_channel_id": "int8",
            "price": "float32",
        }
        transactions = transactions.astype(dtypes_dict)
        transactions.sort_values(["t_dat", "customer_id", "article_id"], inplace=True)
        logger.debug(f"Shape of cleaned transactions data: {transactions.shape}")
        logger.debug(
            f"Size of cleaned transactions data: {human_readable_size(transactions.memory_usage(deep=True).sum())}"
        )
        return transactions

    def _clean_customers_data(self, customers: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning customers data")
        # Clean customer ID from string to int
        customers["customer_id"] = customer_hex_id_to_int(customers["customer_id"])

        # Fill NA
        fillna_dict = {
            "FN": 0,
            "Active": 0,
        }
        customers.fillna(fillna_dict, inplace=True)

        dtypes_dict = {"FN": "int8", "Active": "int8"}
        customers = customers.astype(dtypes_dict)
        customers.sort_values("customer_id", inplace=True)
        logger.debug(f"Shape of cleaned customers data: {customers.shape}")
        logger.debug(f"Size of cleaned customers data: {human_readable_size(customers.memory_usage(deep=True).sum())}")
        return customers

    def _clean_articles_data(self, articles: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning articles data")
        # Clean article ID from string to int
        articles["article_id"] = article_id_str_to_int(articles["article_id"])

        # Change dtype from int64 to int32 for memory optimization
        for col in articles.columns:
            if articles[col].dtype == "int64":
                articles[col] = articles[col].astype("int32")

        articles.sort_values("article_id", inplace=True)
        logger.debug(f"Shape of cleaned articles data: {articles.shape}")
        logger.debug(f"Size of cleaned articles data: {human_readable_size(articles.memory_usage(deep=True).sum())}")
        return articles

    def _subsample_data(
        self, transactions: pd.DataFrame, customers: pd.DataFrame, articles: pd.DataFrame, subsample
    ) -> pd.DataFrame:
        logger.info("Subsampling data")
        if subsample <= 0 or subsample > 1:
            raise ValueError("Subsample must be between 0 and 1 (exclusive).")

        # Subsample customers
        customers_sample = customers.sample(frac=subsample, replace=False, random_state=self.seed)
        customers_sample_ids = set(customers_sample["customer_id"])

        # Subsample transactions and articles based on the sampled customers
        transactions_sample = transactions[transactions["customer_id"].isin(customers_sample_ids)].copy()
        articles_sample_ids = set(transactions_sample["article_id"])
        articles_sample = articles[articles["article_id"].isin(articles_sample_ids)].copy()

        # Reset index
        transactions_sample.reset_index(drop=True, inplace=True)
        customers_sample.reset_index(drop=True, inplace=True)
        articles_sample.reset_index(drop=True, inplace=True)

        logger.debug(f"Shape of subsampled transactions data: {transactions_sample.shape}")
        logger.debug(f"Shape of subsampled customers data: {customers_sample.shape}")
        logger.debug(f"Shape of subsampled articles data: {articles_sample.shape}")
        logger.debug(
            f"Size of subsampled transactions data: {human_readable_size(transactions_sample.memory_usage(deep=True).sum())}"
        )
        logger.debug(
            f"Size of subsampled customers data: {human_readable_size(customers_sample.memory_usage(deep=True).sum())}"
        )
        logger.debug(
            f"Size of subsampled articles data: {human_readable_size(articles_sample.memory_usage(deep=True).sum())}"
        )
        return transactions_sample, customers_sample, articles_sample

    def _train_valid_test_split(self, transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        transactions_test = transactions[transactions["t_dat"] >= TEST_START_DATE].copy()
        transactions_valid = transactions[
            (transactions["t_dat"] < TEST_START_DATE) & (transactions["t_dat"] >= VALID_START_DATE)
        ].copy()
        transactions_train = transactions[transactions["t_dat"] < VALID_START_DATE].copy()

        # Reset index
        transactions_test.reset_index(drop=True, inplace=True)
        transactions_valid.reset_index(drop=True, inplace=True)
        transactions_train.reset_index(drop=True, inplace=True)
        logger.debug(f"Shape of transactions test data: {transactions_test.shape}")
        logger.debug(f"Shape of transactions valid data: {transactions_valid.shape}")
        logger.debug(f"Shape of transactions train data: {transactions_train.shape}")
        logger.debug(
            f"Size of transactions test data: {human_readable_size(transactions_test.memory_usage(deep=True).sum())}"
        )
        logger.debug(
            f"Size of transactions valid data: {human_readable_size(transactions_valid.memory_usage(deep=True).sum())}"
        )
        logger.debug(
            f"Size of transactions train data: {human_readable_size(transactions_train.memory_usage(deep=True).sum())}"
        )

        return transactions_train, transactions_valid, transactions_test

    def _create_customer_article_mapping(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Create a mapping of customer IDs to article IDs based on transactions.
        This is useful for ensuring that articles are linked to the correct customers.
        """
        logger.info("Creating customer-article mapping")
        customer_article_mapping = transactions.groupby("customer_id")["article_id"].apply(list).to_dict()
        logger.debug(f"Len of customer-article mapping: {len(customer_article_mapping)}")
        return customer_article_mapping

    def _save_data(
        self,
        transactions_train: pd.DataFrame,
        transactions_valid: pd.DataFrame,
        transactions_test: pd.DataFrame,
        customer_article_mapping_valid: dict,
        customer_article_mapping_test: dict,
        customers: pd.DataFrame,
        articles: pd.DataFrame,
    ):
        logger.info("Saving processed data")
        # Paths for storing data
        path_transactions_train = self.get_output_paths("transactions", "train", self.subsample, self.seed)
        path_transactions_valid = self.get_output_paths("transactions", "valid", self.subsample, self.seed)
        path_transactions_test = self.get_output_paths("transactions", "test", self.subsample, self.seed)
        path_customers = self.get_output_paths("customers", subsample=self.subsample, seed=self.seed)
        path_articles = self.get_output_paths("articles", subsample=self.subsample, seed=self.seed)

        # Save data
        transactions_train.to_parquet(path_transactions_train)
        transactions_valid.to_parquet(path_transactions_valid)
        transactions_test.to_parquet(path_transactions_test)
        customers.to_parquet(path_customers)
        articles.to_parquet(path_articles)

        # Save customer-article mappings as JSON
        path_mapping_valid = self.get_output_paths("candidates_to_articles_mapping", "valid", self.subsample, self.seed)
        path_mapping_test = self.get_output_paths("candidates_to_articles_mapping", "test", self.subsample, self.seed)
        with open(path_mapping_valid, "w") as f:
            json.dump(customer_article_mapping_valid, f)
        with open(path_mapping_test, "w") as f:
            json.dump(customer_article_mapping_test, f)

        logger.debug(f"Saved transactions train data to {path_transactions_train}")
        logger.debug(f"Saved transactions valid data to {path_transactions_valid}")
        logger.debug(f"Saved transactions test data to {path_transactions_test}")
        logger.debug(f"Saved customers data to {path_customers}")
        logger.debug(f"Saved articles data to {path_articles}")
        logger.info("Data saved successfully")

    def load_and_optimize(self) -> pd.DataFrame:
        """Load and optimize raw data from the H&M dataset."""
        logger.info("Loading and optimizing raw data")
        # Load raw data
        transactions, customers, articles = self._read_raw_data()

        # Clean data
        transactions = self._clean_transactions_data(transactions)
        customers = self._clean_customers_data(customers)
        articles = self._clean_articles_data(articles)

        # Subsample data if required
        if self.subsample < 1:
            transactions, customers, articles = self._subsample_data(transactions, customers, articles, self.subsample)

        # Split transactions data into train, valid, and test sets
        transactions_train, transactions_valid, transactions_test = self._train_valid_test_split(transactions)

        # Create customer-article mapping
        customer_article_mapping_valid = self._create_customer_article_mapping(transactions_valid)
        customer_article_mapping_test = self._create_customer_article_mapping(transactions_test)

        # Save data to parquet files
        self._save_data(
            transactions_train,
            transactions_valid,
            transactions_test,
            customer_article_mapping_valid,
            customer_article_mapping_test,
            customers,
            articles,
        )
