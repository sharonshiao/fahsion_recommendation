import numpy as np
import pandas as pd

DEFAULT_TRAIN_END_DATE = pd.to_datetime("2020-09-08")
DEFAULT_NUM_TRAIN_WEEKS = 6
DEFAULT_TRAIN_START_DATE = DEFAULT_TRAIN_END_DATE - pd.DateOffset(days=DEFAULT_NUM_TRAIN_WEEKS * 7 - 1)
DEFAULT_SEED = 42
DEFAULT_SUBSAMPLE = 0.05

# ======================================================================================================================
# Config for data processing
# ======================================================================================================================
DEFAULT_CUSTOMER_STATIC_FEATURES_CONFIG = {
    "config_processor": {
        "age_bins": [-np.inf, 18, 25, 35, 45, 55, 65, np.inf],
        "keep_numeric_age": False,
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
        "start_week_num": 76,
        "end_week_num": 102,
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
        "start_week_num": 52,
        "end_week_num": 104,
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
    "n_sample_week_threshold_history": -1,
    "negative_sample_strategies": {
        "popularity": {
            "top_k_items": 12,
        },
        "repurchase": {},
    },
    "inference_sample_strategies": {
        "popularity": {
            "top_k_items": 30,
        },
        "repurchase": {},
    },
    "subsample": DEFAULT_SUBSAMPLE,
    "seed": DEFAULT_SEED,
}


DEFAULT_LIGHTGBM_DATA_PROCESSOR_TRAIN_CONFIG = {
    "sample": "train",
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
DEFAULT_RANKER_PIPELINE_CONFIG = {
    "sample": "train",
    "subsample": DEFAULT_SUBSAMPLE,
    "seed": DEFAULT_SEED,
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
            "age_bin",
            "club_member_status",
            "fashion_news_frequency",
            "fn",
            "active",
            "postal_code",
        ],
        "transaction_features": [
            "bestseller_rank",
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
            "random_state": 123,
        },
        "fit_params": {"early_stopping_rounds": 3},
        "use_validation_set": False,
        "save_model": True,
    },
}


# ======================================================================================================================
# Config for model evaluation
# ======================================================================================================================
# config_evaluator_pipeline = {
#     "config_evaluator": {
#         "k": 12,
#     },
#     "test_data": {
#         "data_path": "../data/model/input_inference/valid/subsample_0.05_42",
#         "mapping_path": "../data/candidates_to_articles_mapping_valid_sample_0.05_42.json",
#     },
#     "ranker_path": "../model/lgbm/subsample_0.05_42/1748573531",
# }


def generate_config_evaluator_pipeline(ranker_path: str, sample: str, subsample: float, seed: int) -> dict:
    """Generate config for evaluator pipeline."""
    if sample not in ["valid", "test"]:
        raise ValueError(f"Invalid sample: {sample}")

    data_path = f"../data/model/input_inference/{sample}"
    if subsample < 1:
        data_path += f"/subsample_{subsample}_{seed}"
        mapping_path = f"../data/candidates_to_articles_mapping_{sample}_sample_{subsample}_{seed}.json"
    else:
        data_path += f"/full"
        mapping_path = f"../data/candidates_to_articles_mapping_{sample}_sample_full.json"

    return {
        "config_evaluator": {
            "k": 12,
        },
        "test_data": {
            "data_path": data_path,
            "mapping_path": mapping_path,
        },
        "ranker_path": ranker_path,
    }
