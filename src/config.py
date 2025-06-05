import numpy as np
import pandas as pd

DEFAULT_TRAIN_END_DATE = pd.to_datetime("2020-09-08")
DEFAULT_NUM_TRAIN_WEEKS = 10
DEFAULT_TRAIN_START_DATE = DEFAULT_TRAIN_END_DATE - pd.DateOffset(days=DEFAULT_NUM_TRAIN_WEEKS * 7 - 1)
DEFAULT_SEED = 42
DEFAULT_SUBSAMPLE = 0.25
DEFAULT_TEST_WEEK_NUM = 104
DEFAULT_VALID_WEEK_NUM = 103
DEFAULT_TRAIN_END_WEEK_NUM = 102
DEFAULT_TRAIN_START_WEEK_NUM = DEFAULT_TRAIN_END_WEEK_NUM - DEFAULT_NUM_TRAIN_WEEKS + 1
DEFAULT_HISTORY_START_WEEK_NUM = 52
DEFAULT_HISTORY_START_DATE = DEFAULT_TRAIN_START_DATE - pd.DateOffset(days=7 * 26)
DEFAULT_HISTORY_END_WEEK_NUM = 104

EXPERIMENT_NAME = "fashion_recommendation"

# ======================================================================================================================
# Config for data processing
# ======================================================================================================================
DEFAULT_CUSTOMER_STATIC_FEATURES_CONFIG = {
    "config_processor": {
        "age_bins": [-np.inf, 18, 25, 35, 45, 55, 65, np.inf],
        "keep_numeric_age": True,
        "missing_value_strategy": "fill_unknown",
        "missing_values_map": {
            "fn": 0,
            "active": 0,
            "club_member_status": "unknown",
            "fashion_news_frequency": "unknown",
            "postal_code": "unknown",
        },
        "encoding_strategy": "ordinal",
        "categorical_features": ["club_member_status", "fashion_news_frequency", "postal_code", "age_bin"],
        "numerical_features": ["fn", "active", "age"],
        "one_hot_features": [],
    },
    "subsample": DEFAULT_SUBSAMPLE,
    "seed": DEFAULT_SEED,
}

DEFAULT_CUSTOMER_DYNAMIC_FEATURES_CONFIG = {
    "config_processor": {
        "history_start_week_num": DEFAULT_HISTORY_START_WEEK_NUM,
        "history_end_week_num": DEFAULT_HISTORY_END_WEEK_NUM,
        "start_week_num": DEFAULT_TRAIN_START_WEEK_NUM - 1,
        "end_week_num": DEFAULT_TEST_WEEK_NUM,
        "k_items": 5,
    },
    "subsample": DEFAULT_SUBSAMPLE,
    "seed": DEFAULT_SEED,
}


DEFAULT_ARTICLE_STATIC_FEATURES_CONFIG = {
    "config_processor": {
        "encoding_strategy": "ordinal",
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
    },
    "subsample": DEFAULT_SUBSAMPLE,
    "seed": DEFAULT_SEED,
}

DEFAULT_ARTICLE_DYNAMIC_FEATURES_CONFIG = {
    "config_processor": {
        "encoding_strategy": "ordinal",
        "categorical_features": [],
        "numerical_features": [
            "weekly_sales_count",
            "weekly_avg_price",
            "cumulative_mean_age",
            "cumulative_sales_count",
        ],
        "one_hot_features": [],
        "start_week_num": DEFAULT_TRAIN_START_WEEK_NUM - 1,
        "end_week_num": DEFAULT_TEST_WEEK_NUM,
        "history_start_week_num": DEFAULT_HISTORY_START_WEEK_NUM,
        "history_end_week_num": DEFAULT_HISTORY_END_WEEK_NUM,
    },
    "subsample": DEFAULT_SUBSAMPLE,
    "seed": DEFAULT_SEED,
}

DEFAULT_ARTICLE_EMBEDDING_CONFIG = {
    "config_processor": {
        "device_type": "mps",
        "batch_size": 32,
        "text_model_id": "distilbert-base-uncased",
        "img_model_id": "resnet18",
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
    },
    "subsample": 1.0,
    "seed": DEFAULT_SEED,
}


DEFAULT_CANDIDATE_GENERATION_CONFIG = {
    "train_start_date": DEFAULT_TRAIN_START_DATE,
    "train_end_date": DEFAULT_TRAIN_END_DATE,
    "history_start_date": DEFAULT_HISTORY_START_DATE,
    "n_sample_week_threshold": -1,
    "negative_sample_strategies": {
        "popularity": {
            "top_k_items": 30,
        },
        "repurchase": {
            "strategy": "last_k_items",
            "k": 12,
        },
    },
    "inference_sample_strategies": {
        "popularity": {
            "top_k_items": 30,
        },
        "repurchase": {
            "strategy": "last_k_items",
            "k": 12,
        },
    },
    "subsample": DEFAULT_SUBSAMPLE,
    "seed": DEFAULT_SEED,
    "restrict_positive_samples": True,
    "neg_to_pos_ratio": 30.0,
}


DEFAULT_LIGHTGBM_DATA_PROCESSOR_TRAIN_CONFIG = {
    "sample": "train",
    "include_article_static_features": True,
    "include_article_dynamic_features": True,
    "include_customer_static_features": True,
    "include_customer_dynamic_features": True,
    "include_transaction_features": True,
    "include_user_history": False,
    "use_default_data_paths": True,
    "subsample": DEFAULT_SUBSAMPLE,
    "seed": DEFAULT_SEED,
    "data_paths": {"candidates": "", "article_features": "", "customer_features": ""},
}

DEFAULT_LIGHTGBM_DATA_PROCESSOR_VALID_CONFIG = {
    "sample": "valid",
    "include_article_static_features": True,
    "include_article_dynamic_features": True,
    "include_customer_static_features": True,
    "include_transaction_features": True,
    "include_user_history": False,
    "use_default_data_paths": True,
    "subsample": DEFAULT_SUBSAMPLE,
    "seed": DEFAULT_SEED,
    "data_paths": {"candidates": "", "article_features": "", "customer_features": ""},
}

DEFAULT_LIGHTGBM_DATA_PROCESSOR_TEST_CONFIG = {
    "sample": "test",
    "include_article_static_features": True,
    "include_article_dynamic_features": True,
    "include_customer_static_features": True,
    "include_transaction_features": True,
    "include_user_history": False,
    "use_default_data_paths": True,
    "subsample": DEFAULT_SUBSAMPLE,
    "seed": DEFAULT_SEED,
    "data_paths": {"candidates": "", "article_features": "", "customer_features": ""},
}

# ======================================================================================================================
# Config for model training
# ======================================================================================================================
# This is for adhoc training
DEFAULT_RANKER_PIPELINE_CONFIG = {
    "sample": "train",
    "subsample": DEFAULT_SUBSAMPLE,
    "seed": DEFAULT_SEED,
    "tag": "testing_features",
    "experiment_name": EXPERIMENT_NAME,
    "feature_config": {
        # Feature lists by domain
        "article_static_features": [
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
        "article_dynamic_features": [
            "weekly_sales_count",
            "weekly_avg_price",
            "cumulative_mean_age",
            "cumulative_sales_count",
        ],
        "customer_static_features": [
            # "age_bin",
            "age",
            "club_member_status",
            "fashion_news_frequency",
            "fn",
            "active",
            "postal_code",
        ],
        "customer_dynamic_features": [
            "customer_avg_price",
            "text_embedding_similarity",
            "image_embedding_similarity",
        ],
        "transaction_features": [
            # "price",
            "bestseller_rank",
        ],
        "interactions": [
            "age_difference",
            "age_ratio",
            "price_difference",
            "price_ratio",
        ],
    },
    "lightgbm_params": {
        "ranker_params": {
            "objective": "lambdarank",
            "metrics": "ndcg",
            "boosting_type": "dart",
            "n_estimators": 1,
            "importance_type": "gain",
            "verbose": 10,
            "random_state": 111,
        },
        "fit_params": {"early_stopping_rounds": 3},
        "use_validation_set": True,
        "save_model": True,
    },
}


DEFAULT_RANKER_HYPERPARAMETERS_PIPELINE_CONFIG = {
    # I/O
    "sample": "train",
    "subsample": DEFAULT_SUBSAMPLE,
    "seed": DEFAULT_SEED,
    "tag": "ranker-hyperparameter-tuning",
    "experiment_name": EXPERIMENT_NAME,
    # Parameters with hyperparameter tuning
    "hyperparameter_config": {
        "feature_config": {
            # Feature lists by domain
            "article_static_features": [
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
            "article_dynamic_features": [
                # "weekly_sales_count",
                # "weekly_avg_price",
                # "cumulative_mean_age",
                # "cumulative_sales_count",
            ],
            "customer_static_features": [
                # "age_bin",
                "age",
                "club_member_status",
                "fashion_news_frequency",
                "fn",
                "active",
                "postal_code",
            ],
            "customer_dynamic_features": [
                # "customer_avg_price",
                # "text_embedding_similarity",
                # "image_embedding_similarity",
            ],
            "transaction_features": [
                # "price",
                # "bestseller_rank",
            ],
            "interactions": [
                # "age_difference",
                # "age_ratio",
                # "price_difference",
                # "price_ratio",
            ],
        },
        "lightgbm_fixed_params": {
            "objective": "lambdarank",
            "metrics": "ndcg",
            # "boosting_type": "dart",
            "importance_type": "gain",
            "random_state": 111,
            "verbosity": -1,
            "feature_pre_filter": False,
        },
        "hyperparameters_config": {
            "learning_rate": {
                "type": "float",
                "min": 1e-5,
                "max": 0.2,
            },
            "n_estimators": {
                "type": "int",
                "min": 50,
                "max": 500,
            },
            "reg_alpha": {
                "type": "float",
                "min": 1e-8,
                "max": 100,
                "log": True,
            },
            "reg_lambda": {
                "type": "float",
                "min": 1e-8,
                "max": 100,
                "log": True,
            },
            "num_leaves": {
                "type": "int",
                "min": 2,
                "max": 512,
            },
            "colsample_bytree": {
                "type": "float",
                "min": 0.1,
                "max": 1.0,
            },
            "subsample": {
                "type": "float",
                "min": 0.01,
                "max": 1.0,
            },
            "subsample_freq": {
                "type": "int",
                "min": 1,
                "max": 7,
            },
            "min_child_samples": {
                "type": "int",
                "min": 5,
                "max": 100,
            },
        },
        # Hyperparams
        # "learning_rate": trial.suggest_float("learning_rate", lr_min, lr_max),
        # "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        # "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100, log=True),
        # "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100, log=True),
        # "num_leaves": trial.suggest_int("num_leaves", 2, 512),
        # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        # "subsample": trial.suggest_float("subsample", 0.01, 1.0),
        # "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
        # "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "n_trials": 50,
        "early_stopping_rounds": 25,
        "metric_return": "mapk_valid_inference",
    },
}

# THis is the mlflow run id used for hyperparameter tuning
# This will be used for training the model with tuned hyperparameters
# RUN_ID_HYPERPARAMETER_TUNING = "892dc8321cf34c689440de0d536f72f7" # Base, for testing pipeline
RUN_ID_HYPERPARAMETER_TUNING = "afbd31fb97b3437a89023e5587d02d3a"


# Run if for the tuned ranker
# RUN_ID_TUNED_xRANKER = "bd823ea190a7481bbdb932be52d4cb8b"  # Base, for testing pipeline
RUN_ID_TUNED_RANKER = "afe1c7386b374e3d9b11588803db5ee0"

DEFAULT_RANKER_EVALUATOR_CONFIG = {
    # I/O
    "sample": ["valid", "test"],
    "subsample": DEFAULT_SUBSAMPLE,
    "seed": DEFAULT_SEED,
    # Evaluation parameters
    "config_evaluator": {
        "k": 12,
        "heuristic_strategy": "rolling_popular_items",
    },
    # Experiment tracking
    "run_id": RUN_ID_TUNED_RANKER,
    "experiment_name": EXPERIMENT_NAME,
}
