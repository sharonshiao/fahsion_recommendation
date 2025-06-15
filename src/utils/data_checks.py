from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.metrics.pairwise import cosine_similarity

from src.candidate_generator import (
    CandidateGeneratorPipelineConfig,
    CandidateGeneratorResult,
    week_num_from_week,
)
from src.feature_customers import (
    CustomerDynamicFeaturePipelineConfig,
    CustomerDynamicFeatureResult,
)
from src.features_articles import (
    ArticleDynamicFeaturePipelineConfig,
    ArticleDynamicFeatureResult,
    ArticleEmbeddingResult,
)
from src.input_preprocessing import LightGBMDataResult
from src.sampling.manager import WEEK_NUM_TEST, WEEK_NUM_VALID


def test_feature_article_dynamic_feature_pipeline(
    results_articles_dynamic: ArticleDynamicFeatureResult,
    articles: pd.DataFrame,
    pipeline_config: ArticleDynamicFeaturePipelineConfig,
):
    start_week_num = pipeline_config.config_processor["start_week_num"]
    end_week_num = pipeline_config.config_processor["end_week_num"]

    # Check min week and max week
    assert (
        results_articles_dynamic.data.week_num.min() == start_week_num
    ), "Minimum week number should be equal to the start week number"
    assert (
        results_articles_dynamic.data.week_num.max() == end_week_num
    ), "Maximum date should be equal to the maximum date in the articles dataframe"

    # Checking that shapes and dates are correct
    assert (
        results_articles_dynamic.data.article_id.nunique() == articles.shape[0]
    ), "Number of articles should be equal to the number of unique articles in the articles dataframe"
    assert (
        results_articles_dynamic.data.week_num.min() == start_week_num
    ), "Minimum week number should be equal to the start week number"
    assert (
        results_articles_dynamic.data.week_num.max() == end_week_num
    ), "Maximum week number should be equal to the end week number"

    # Number of rows = number of articles * number of weeks
    assert results_articles_dynamic.data.shape[0] == articles.shape[0] * (
        end_week_num - start_week_num + 1
    ), "Number of rows should be equal to the number of articles * number of weeks"

    print("Metadata checks passed")


def test_feature_article_dynamic_feature(
    transactions: pd.DataFrame,
    results_articles_dynamic: ArticleDynamicFeatureResult,
    customers: pd.DataFrame,
    pipeline_config: ArticleDynamicFeaturePipelineConfig,
    article_ids: List[int],
):
    start_week_num = pipeline_config.config_processor["start_week_num"]  # noqa: F841
    end_week_num = pipeline_config.config_processor["end_week_num"]  # noqa: F841
    history_start_week_num = pipeline_config.config_processor["history_start_week_num"]  # noqa: F841

    for article_id in article_ids:

        print("Article ID", article_id)

        # Get weeks with sales record for this article
        weeks = transactions.query(
            "article_id == @article_id and week_num >= @start_week_num and week_num <= @end_week_num"
        ).week_num.unique()

        # For each week where there is a sales record, check that the features are equal to the expected values
        for week_num in weeks:
            print("Week number", week_num)
            # Calculate expected values from transactions
            expected = (
                transactions.query(
                    "week_num >= @history_start_week_num and week_num <= @week_num and article_id == @article_id"
                )
                .merge(customers, on="customer_id")
                .agg({"age": "mean", "customer_id": "count"})
            )

            stats = transactions.query("week_num == @week_num and article_id == @article_id").agg(
                {"price": "mean", "customer_id": "count"}
            )

            res = results_articles_dynamic.data.query("article_id == @article_id and week_num == @week_num")

            assert np.isclose(
                expected.age, res.cumulative_mean_age.values[0]
            ), f"Age is not equal for article {article_id} and week {week_num}. Expected {expected.age}, got {res.cumulative_mean_age.values[0]}"
            assert np.isclose(
                expected.customer_id, res.cumulative_sales_count.values[0]
            ), f"Customer ID is not equal for article {article_id} and week {week_num}. Expected {expected.customer_id}, got {res.cumulative_sales_count.values[0]}"
            assert np.isclose(
                stats.price, res.weekly_avg_price.values[0]
            ), f"Price is not equal for article {article_id} and week {week_num}. Expected {stats.price}, got {res.weekly_avg_price.values[0]}"
            assert np.isclose(
                stats.customer_id, res.weekly_sales_count.values[0]
            ), f"Customer ID is not equal for article {article_id} and week {week_num}. Expected {stats.customer_id}, got {res.weekly_sales_count.values[0]}"
            print("Numbers matched for week", week_num)

        print("-" * 80)


def test_feature_customer_avg_embedding_pipeline(
    results_customer_dynamic_feature: CustomerDynamicFeatureResult,
    transactions: pd.DataFrame,
    pipeline_config: CustomerDynamicFeaturePipelineConfig,
):
    """Test the metadata of the customer dynamic feature pipeline."""
    # There should be equal number of rows per week
    assert (
        results_customer_dynamic_feature.data.groupby("week_num").size().nunique() == 1
    ), "There should be equal number of rows per week"

    # Number of customers should equal the number of unique customers in transactions in the given period
    expected_num_customers = transactions.query(
        "week_num >= @pipeline_config.config_processor['start_week_num'] and week_num <= @pipeline_config.config_processor['end_week_num']"
    ).customer_id.nunique()
    assert (
        results_customer_dynamic_feature.data.customer_id.nunique() == expected_num_customers
    ), f"Number of customers should equal the number of unique customers in transactions in the given period. Expected {expected_num_customers}, got {results_customer_dynamic_feature.data.customer_id.nunique()}"

    # Number of weeks should equal the number of unique weeks in transactions in the given period
    expected_num_weeks = transactions.query(
        "week_num >= @pipeline_config.config_processor['start_week_num'] and week_num <= @pipeline_config.config_processor['end_week_num']"
    ).week_num.nunique()
    assert (
        results_customer_dynamic_feature.data.week_num.nunique() == expected_num_weeks
    ), "Number of weeks should equal the number of unique weeks in transactions in the given period"

    # Check that the number of rows is equal to the number of customers * number of weeks
    assert (
        results_customer_dynamic_feature.data.shape[0] == expected_num_customers * expected_num_weeks
    ), "Number of rows should equal the number of customers * number of weeks"

    print("Checks passed")


def test_feature_customer_avg_embedding(
    results_customer_dynamic_feature: CustomerDynamicFeatureResult,
    transactions: pd.DataFrame,
    results_article_embeddings: ArticleEmbeddingResult,
    pipeline_config: CustomerDynamicFeaturePipelineConfig,
    customer_id: int,
):

    history_start_week_num = pipeline_config.config_processor["history_start_week_num"]  # noqa: F841
    weeks = results_customer_dynamic_feature.data.query("customer_id == @customer_id").week_num.unique()
    print(f"Customer {customer_id} has {len(weeks)} weeks: {weeks}")

    for week_num in weeks:
        # Get the avg embedding we calculate from the processor
        res = results_customer_dynamic_feature.data.query("customer_id == @customer_id and week_num == @week_num")
        res_price = res.customer_avg_price.values[0]
        res_embeddings = res.customer_avg_text_embedding.values[0]

        # Check the raw data
        tmp = (
            transactions.query(
                "customer_id == @customer_id and week_num <= @week_num and week_num >= @history_start_week_num"
            )
            .sort_values(["t_dat", "article_id"], ascending=[False, True])
            .drop_duplicates(subset=["article_id"])
        ).head(5)
        display(tmp)

        if tmp.shape[0] == 0:
            print(f"No previous transactions for customer {customer_id} and week {week_num}")
            continue

        print(f"Calculating avg price and embeddings for customer {customer_id} and week {week_num}")
        avg_price = tmp.price.mean()
        avg_embeddings = np.mean(
            [
                results_article_embeddings.text_embeddings[results_article_embeddings.id_to_index[v]]
                for v in tmp.article_id
            ],
            axis=0,
        )

        assert (
            avg_embeddings.shape == res_embeddings.shape
        ), f"Avg embeddings are not equal for customer {customer_id} and week {week_num}"
        assert np.allclose(
            avg_price, res_price
        ), f"Avg price is not equal for customer {customer_id} and week {week_num}"
        assert np.allclose(
            avg_embeddings, res_embeddings
        ), f"Avg embeddings are not equal for customer {customer_id} and week {week_num}"
        print(f"Checks for customer {customer_id} and week {week_num} passed")


def test_candidate_generator_pipeline(
    results_candidate_generation: CandidateGeneratorResult,
    transactions: pd.DataFrame,
    pipeline_config: CandidateGeneratorPipelineConfig,
):
    has_data = results_candidate_generation.data is not None
    sample = results_candidate_generation.sample
    print(f"Has data: {has_data}, sample: {sample}")

    # Check that the week range is correct
    if sample == "train":
        start_week_num = week_num_from_week(pipeline_config.train_start_date)
        end_week_num = week_num_from_week(pipeline_config.train_end_date)
    elif sample == "valid":
        start_week_num = WEEK_NUM_VALID
        end_week_num = WEEK_NUM_VALID
    elif sample == "test":
        start_week_num = WEEK_NUM_TEST
        end_week_num = WEEK_NUM_TEST

    print("Checking week range")
    if has_data and sample == "train":

        assert (
            results_candidate_generation.data.week_num.min() == start_week_num
        ), "Week number should be equal to the train start date"
        assert (
            results_candidate_generation.data.week_num.max() == end_week_num
        ), "Week number should be equal to the train end date"

    elif has_data and sample == "validation":
        assert (
            results_candidate_generation.data.week_num.min() == start_week_num
        ), "Week number should be equal to the validation start date"
        assert (
            results_candidate_generation.data.week_num.max() == end_week_num
        ), "Week number should be equal to the validation end date"

    elif has_data and sample == "test":
        assert (
            results_candidate_generation.data.week_num.min() == start_week_num
        ), "Week number should be equal to the test start date"
        assert (
            results_candidate_generation.data.week_num.max() == end_week_num
        ), "Week number should be equal to the test end date"

    # Check that default prediction is avaiable if data_type == "inference"
    print("Checking default prediction")
    if sample != "train":
        assert results_candidate_generation.default_prediction is not None, "Default prediction should be available"
    else:
        assert results_candidate_generation.default_prediction is None, "Default prediction should not be available"

    # Checking number of customers
    print("Checking number of customers")
    expected_num_customers = transactions.query(
        "week_num >= @start_week_num and week_num <= @end_week_num"
    ).customer_id.nunique()
    assert (
        results_candidate_generation.data.customer_id.nunique() == expected_num_customers
    ), "Number of customers should equal the number of unique customers in transactions in the given period"

    print("Checks passed")


class CandidateGeneratorTest:

    def __init__(self):
        pass

    @staticmethod
    def test_positive_examples(
        results_candidate_generation: CandidateGeneratorResult,
        transactions: pd.DataFrame,
        pipeline_config: CandidateGeneratorPipelineConfig,
    ):
        print("Testing positive examples")
        # For train dataset, test that all positive examples are in transactions
        positive_examples = results_candidate_generation.data.query("label == 1")
        join = positive_examples.merge(
            transactions, on=["customer_id", "article_id", "week_num"], how="left", indicator=True
        )
        assert join["_merge"].unique() == ["both"], "All positive examples should be in transactions"

        if pipeline_config.restrict_positive_samples:
            # Check that there should be no positive sources
            assert (
                results_candidate_generation.data.query("label == 1 and source == 'positive'").shape[0] == 0
            ), "There should be no positive sources"

    @staticmethod
    def test_negative_examples(results_candidate_generation: CandidateGeneratorResult, transactions: pd.DataFrame):
        print("Testing negative examples")
        negative_examples = results_candidate_generation.data.query("label == 0")
        join = negative_examples.merge(
            transactions, on=["customer_id", "article_id", "week_num"], how="left", indicator=True
        )
        assert join["_merge"].value_counts().loc["both"] == 0, "All negative examples should not be in transactions"

    @staticmethod
    def test_repurchases_last_purchase(
        results_candidate_generation: CandidateGeneratorResult,
        transactions: pd.DataFrame,
        pipeline_config: CandidateGeneratorPipelineConfig,
        customer_ids: List[int],
    ):
        """Check that the last purchase is being added as a candidate for the next purchase week."""
        print("Testing repurchases - last purchase")
        train_start_date = week_num_from_week(pipeline_config.train_start_date)  # noqa: F841
        sample = results_candidate_generation.sample  # noqa: F841
        if sample == "train":
            train_end_date = week_num_from_week(pipeline_config.train_end_date) - 1  # noqa: F841
        elif sample == "valid":
            train_end_date = WEEK_NUM_VALID - 1  # noqa: F841
        else:
            train_end_date = WEEK_NUM_TEST - 1  # noqa: F841

        for customer_id in customer_ids:
            print(f"Customer {customer_id}")

            # For each customer, get the weeks with transactions used for calculating repurchases
            weeks = transactions.query(
                "customer_id == @customer_id and week_num >= @train_start_date and week_num <= @train_end_date"
            ).week_num.unique()
            weeks.sort()

            if results_candidate_generation.sample != "train":
                weeks = weeks[-1:]
            print(f"Customer {customer_id} has {len(weeks)} weeks: {weeks}")

            for i, week_num in enumerate(sorted(weeks)):
                print(f"Week {week_num}")
                res = transactions.query("customer_id == @customer_id and week_num == @week_num")
                if i < len(weeks) - 1:
                    next_purchase_week_num = weeks[i + 1]
                elif results_candidate_generation.sample == "test":
                    next_purchase_week_num = WEEK_NUM_TEST
                elif results_candidate_generation.sample == "valid":
                    next_purchase_week_num = WEEK_NUM_VALID
                else:
                    # For train, we don't have a next purchase week
                    print(
                        f"No next purchase week for customer {customer_id} and week {week_num} because it's the last week in train dataset"
                    )
                    continue

                # Check that each article is in the candidate list in week_num + 1
                print(f"Checking {len(res)} article_ids in week {next_purchase_week_num}")
                for article_id in res.article_id:
                    assert (
                        article_id
                        in results_candidate_generation.data.query(
                            "customer_id == @customer_id and week_num == @next_purchase_week_num"
                        ).article_id.unique()
                    ), f"Article {article_id} is not in the candidate list for customer {customer_id} and week {next_purchase_week_num}"
                print(f"Checks for customer {customer_id} and week {week_num} passed")
                print("-" * 80)

            print("=" * 80)
            print("")

    @staticmethod
    def test_repurchases_last_k_items(
        results_candidate_generation: CandidateGeneratorResult,
        transactions: pd.DataFrame,
        pipeline_config: CandidateGeneratorPipelineConfig,
        customer_ids: List[int],
    ):
        """Check that the last k items are being added as candidates for the next purchase week."""
        print("Testing repurchases - last k items")
        train_start_date = week_num_from_week(pipeline_config.train_start_date)  # noqa: F841
        train_end_date = week_num_from_week(pipeline_config.train_end_date) - 1  # noqa: F841
        history_start_date = week_num_from_week(pipeline_config.history_start_date)  # noqa: F841
        k = pipeline_config.negative_sample_strategies["repurchase"]["k"]

        for customer_id in customer_ids:
            print(f"Customer {customer_id}")

            # For each customer, get the weeks with transactions used for calculating repurchases
            if results_candidate_generation.sample == "train":
                weeks = transactions.query(
                    "customer_id == @customer_id and week_num >= @train_start_date and week_num <= @train_end_date"
                ).week_num.unique()
                weeks.sort()
            elif results_candidate_generation.sample == "valid":
                weeks = [WEEK_NUM_VALID]
            else:
                weeks = [WEEK_NUM_TEST]

            # For each week, get the last k items
            for week_num in weeks:
                print(f"Week {week_num}")
                res = transactions.query(
                    "customer_id == @customer_id and week_num < @week_num and week_num >= @history_start_date"
                ).sort_values(["week_num", "article_id"], ascending=[False, True])
                last_k_items = set(res.head(k).article_id.tolist())

                # Check that the last k items are in the candidate list for week_num
                candidate_set = set(
                    results_candidate_generation.data.query(
                        "customer_id == @customer_id and week_num == @week_num"
                    ).article_id.unique()
                )
                print(f"Candidate set: {candidate_set}")
                print(f"Last {k} items: {last_k_items}")
                print("Diff:", last_k_items - candidate_set)
                assert last_k_items.issubset(
                    candidate_set
                ), f"Last {k} items are not in the candidate list for customer {customer_id} and week {week_num}"
                print(f"Checks for customer {customer_id} and week {week_num} passed")
                print("-" * 80)

            print("=" * 80)
            print("")

    @staticmethod
    def test_repurchases(
        results_candidate_generation: CandidateGeneratorResult,
        transactions: pd.DataFrame,
        pipeline_config: CandidateGeneratorPipelineConfig,
        customer_ids: List[int],
    ):
        if pipeline_config.neg_to_pos_ratio != -1:
            print("Skipping repurchases test because neg_to_pos_ratio is not -1")
            return

        if pipeline_config.negative_sample_strategies["repurchase"]["strategy"] == "last_purchase":
            CandidateGeneratorTest.test_repurchases_last_purchase(
                results_candidate_generation, transactions, pipeline_config, customer_ids
            )
        elif pipeline_config.negative_sample_strategies["repurchase"]["strategy"] == "last_k_items":
            CandidateGeneratorTest.test_repurchases_last_k_items(
                results_candidate_generation, transactions, pipeline_config, customer_ids
            )
        else:
            raise ValueError(
                f"Invalid repurchase strategy: {pipeline_config.negative_sample_strategies['repurchase']['strategy']}"
            )

    @staticmethod
    def test(
        results_candidate_generation: CandidateGeneratorResult,
        transactions: pd.DataFrame,
        pipeline_config: CandidateGeneratorPipelineConfig,
        customer_ids: List[int],
    ):
        """Test that the candidate generator is correctly generating candidates."""
        print("Testing candidate generator")

        sample = results_candidate_generation.sample
        print(f"Sample: {sample}")

        # Test positive examples
        CandidateGeneratorTest.test_positive_examples(results_candidate_generation, transactions, pipeline_config)

        # Test negative examples
        CandidateGeneratorTest.test_negative_examples(results_candidate_generation, transactions)

        # Check repurchases for some customers
        CandidateGeneratorTest.test_repurchases(
            results_candidate_generation, transactions, pipeline_config, customer_ids
        )

        print("Checks passed")


# Keeping for backward compatibility
def test_candidate_generator(
    results_candidate_generation: CandidateGeneratorResult,
    transactions: pd.DataFrame,
    pipeline_config: CandidateGeneratorPipelineConfig,
    customer_ids: List[int],
):
    """Test that the candidate generator is correctly generating candidates."""
    CandidateGeneratorTest.test(results_candidate_generation, transactions, pipeline_config, customer_ids)


def test_input_embedding_similarity(
    results_train: LightGBMDataResult,
    customer_dynamic_features: CustomerDynamicFeatureResult,
    article_embeddings: ArticleEmbeddingResult,
    customer_id: int,
    items_to_test: int = 3,
):
    """This function tests that the embedding similarity is correctly calculated for a given customer. 3
    articles per week are picked and the embedding similarity is checked."""

    # Calculate cosine similarity
    # Find the unique number of weeks for the customer
    unique_week_numbers = results_train.data.query("customer_id == @customer_id").week_num.unique()
    print(
        "Unique week numbers for customer",
        customer_id,
        unique_week_numbers,
    )

    for week_num in unique_week_numbers:
        print("Week number:", week_num)
        print(f"Top {items_to_test} rows:")
        res = results_train.data.query("customer_id == @customer_id and week_num == @week_num")[
            ["week_num", "customer_id", "article_id", "text_embedding_similarity", "image_embedding_similarity"]
        ].head(items_to_test)
        display(res)
        # Get article ids to check
        article_ids = res.article_id.tolist()

        # Get history for the customer and the average embeddings
        history = customer_dynamic_features.data.query("customer_id == @customer_id and week_num == @week_num - 1")
        display(history)
        customer_avg_text_embedding = history.customer_avg_text_embedding.values[0]
        customer_avg_image_embedding = history.customer_avg_image_embedding.values[0]

        # Calculate similarity for each article
        for article_id in article_ids:
            print("Article ID:", article_id)
            article_text_embedding = article_embeddings.text_embeddings[article_embeddings.id_to_index[article_id]]
            article_image_embedding = article_embeddings.image_embeddings[article_embeddings.id_to_index[article_id]]

            if customer_avg_text_embedding is None:
                text_similarity = 0
            else:
                text_similarity = cosine_similarity(
                    customer_avg_text_embedding.reshape(1, -1), article_text_embedding.reshape(1, -1)
                )

            if customer_avg_image_embedding is None:
                image_similarity = 0
            else:
                image_similarity = cosine_similarity(
                    customer_avg_image_embedding.reshape(1, -1), article_image_embedding.reshape(1, -1)
                )

            # Checks
            assert np.isclose(
                text_similarity, res.query("article_id == @article_id").text_embedding_similarity.values[0]
            ), f"Text similarity is not equal for article {article_id} and week {week_num}"
            assert np.isclose(
                image_similarity, res.query("article_id == @article_id").image_embedding_similarity.values[0]
            ), f"Image similarity is not equal for article {article_id} and week {week_num}"

            print("Checks for article", article_id, "passed")


def test_input_customer_avg_price(
    results_train: LightGBMDataResult, customer_dynamic_features: CustomerDynamicFeatureResult, customer_id: int
):
    """Test that the customer avg price is correctly calculated."""

    # Get the customer avg price
    weeks = results_train.data.query("customer_id == @customer_id").week_num.unique()
    print(f"Customer {customer_id} has {len(weeks)} weeks: {weeks}")

    for week_num in weeks:
        print(f"Week number: {week_num}")
        res = results_train.data.query("customer_id == @customer_id and week_num == @week_num")
        customer_avg_price = res.customer_avg_price.unique()
        assert (
            len(customer_avg_price) == 1
        ), f"There should be only one row for customer {customer_id} and week {week_num}"
        assert np.isclose(
            customer_avg_price[0],
            customer_dynamic_features.data.query(
                "customer_id == @customer_id and week_num == @week_num - 1"
            ).customer_avg_price.values[0],
        ), f"Customer avg price is not equal for customer {customer_id} and week {week_num}"
        print(f"Checks for customer {customer_id} and week {week_num} passed")


def test_input_articles_dynamic_features(
    article_dynamic_features: ArticleDynamicFeatureResult, results_train: LightGBMDataResult, items_to_test: int = 3
):
    """Test that article dynamic features are correctly joined with the results. We pick items_to_test articles per week and check
    that the features are equal."""

    for week_num in np.sort(results_train.data.week_num.unique()):

        print(f"Week number: {week_num}")

        # Get items_to_test articles in the week
        tmp_articles_in_week = (
            results_train.data[
                [
                    "customer_id",
                    "week_num",
                    "article_id",
                    "weekly_sales_count",
                    "weekly_avg_price",
                    "cumulative_sales_count",
                    "cumulative_mean_age",
                ]
            ]
            .query("week_num == @week_num")
            .head(items_to_test)
        )

        display(tmp_articles_in_week)

        # Check that features are equal by looking at article_dynamic_features
        for article_id in tmp_articles_in_week.article_id:
            print("Article ID:", article_id)
            article_features = article_dynamic_features.data.query(
                "article_id == @article_id and week_num == @week_num - 1"
            )
            display(article_features)

            # Check that features are equal

            assert np.isclose(
                tmp_articles_in_week[tmp_articles_in_week.article_id == article_id].weekly_sales_count.values[0],
                article_features.weekly_sales_count.values[0],
            ), f"Weekly sales count is not equal for article {article_id} and week {week_num}"
            assert np.isclose(
                tmp_articles_in_week[tmp_articles_in_week.article_id == article_id].weekly_avg_price.values[0],
                article_features.weekly_avg_price.values[0],
            ), f"Weekly avg price is not equal for article {article_id} and week {week_num}"
            assert np.isclose(
                tmp_articles_in_week[tmp_articles_in_week.article_id == article_id].cumulative_sales_count.values[0],
                article_features.cumulative_sales_count.values[0],
            ), f"Cumulative sales count is not equal for article {article_id} and week {week_num}"
            assert np.isclose(
                tmp_articles_in_week[tmp_articles_in_week.article_id == article_id].cumulative_mean_age.values[0],
                article_features.cumulative_mean_age.values[0],
            ), f"Cumulative mean age is not equal for article {article_id} and week {week_num}"

            print(f"Checks for article {article_id} and week {week_num} passed")
            print("-" * 80)


def test_lightgbm_data_pipeline_metadata(results: LightGBMDataResult, candidates: CandidateGeneratorResult):
    """Test metadata of the lightgbm data pipeline."""

    print(f"Results use type: {results.use_type}, candidates sample: {candidates.sample}")

    if results.use_type == "train":
        expected = candidates.data
    else:
        expected = candidates.data_inference

    # Check that the week range is correct
    print("Checking week range")
    assert results.data.week_num.min() == expected.week_num.min()
    assert results.data.week_num.max() == expected.week_num.max()

    # Check that the number of customers is correct
    print("Checking number of customers")
    assert results.data.customer_id.nunique() == expected.customer_id.nunique()

    # Check that the number of articles is correct
    print("Checking number of articles")
    assert results.data.article_id.nunique() == expected.article_id.nunique()

    # Check that the number of rows is correct
    print("Checking number of rows")
    assert results.data.shape[0] == expected.shape[0]

    # Check no duplicate customer_id, article_id, week_num
    print("Checking no duplicate customer_id, article_id, week_num")
    assert (
        results.data.drop_duplicates(subset=["customer_id", "article_id", "week_num"]).shape[0] == results.data.shape[0]
    )

    print("Checks passed")


def test_lightgbm_data_features(results: LightGBMDataResult, verbose: bool = False):
    print(f"Results use type: {results.use_type}, sample: {results.sample}")

    features = results.get_feature_names_list()
    data = results.data

    # Check that there should be no missing values
    print("Checking no missing values")
    assert data.isnull().sum().sum() == 0, f"There should be no missing values, got {data.isnull().sum()}"

    # Check that there should be no duplicate columns
    print("Checking no duplicate columns")
    assert data.columns.duplicated().sum() == 0, "There should be no duplicate columns"

    # Print out feature distribution and show histogram
    if verbose:
        for feature in features:
            print(f"Feature: {feature}")
            print(data[feature].describe())
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(data[feature], ax=ax)
            plt.show()
            print("-" * 80)

    print("Checks passed")
