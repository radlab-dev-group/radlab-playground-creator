import os
import django
import logging
import datetime
import argparse

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from system.controllers import SystemController
from creator.controllers.news import NewsController
from creator.controllers.clustering import ClusteringHandler
from apps_creator.periodic.src.utils import InformationBrowser


def prepare_parser(desc=""):
    p = argparse.ArgumentParser(description=desc)

    p.add_argument(
        "--min-cluster-count", dest="min_cluster_count", type=int, default=25
    )
    p.add_argument(
        "--opt-cluster-count", dest="opt_cluster_count", type=int, default=35
    )
    p.add_argument(
        "--max-cluster-count", dest="max_cluster_count", type=int, default=45
    )

    return p


def main(argv=None):
    args = prepare_parser(argv).parse_args(argv)

    system_settings = SystemController.get_system_settings()
    if system_settings.doing_news_generation_for_yesterday:
        return

    SystemController.begin_public_yesterday_news_generation(system_settings)

    news_controller = NewsController(
        add_to_db=True,
        seconds_prev_check=0,
        models_config_path=None,
    )

    clear_dataset_if_exists = True

    end_date = datetime.datetime.now().date()
    begin_date = end_date - datetime.timedelta(days=1)
    logging.info(f"Generating news for days {begin_date} - {end_date}")

    # Gen news for date range
    gen_news = news_controller.public_get_generated_news_for_date_range(
        begin_date=begin_date, end_date=end_date
    )

    # Convert news to pairs of text and metadata
    conv_news = InformationBrowser.convert_news_to_store_jsonl(
        generated_news=gen_news
    )

    # Store converted news to a temporary file
    news_temp_file = InformationBrowser.store_converted_news_to_jsonl_file(
        all_news=conv_news, out_file_path=None
    )
    if not os.path.exists(news_temp_file):
        logging.error(f"Temporary file {news_temp_file} does not exists!")
        logging.error(f"Cannot continue news generation! Check file existing.")
        SystemController.end_public_yesterday_news_generation(system_settings)
        return

    # Prepare clusterer
    cl_handler = ClusteringHandler(
        clustering_config_path="configs/clusterer-config.json",
        min_cluster_count=args.min_cluster_count,
        opt_cluster_count=args.opt_cluster_count,
        max_cluster_count=args.max_cluster_count,
    )

    # Clear dataset before loading if necessary
    if clear_dataset_if_exists:
        cl_handler.clear()

    # load clusterer dataset from temp file
    cl_handler.clusterer.load_dataset(
        file_path=news_temp_file,
        text_column=InformationBrowser.TEXT_COLUMN_NAME,
        metadata_column=InformationBrowser.METADATA_COLUMN_NAME,
        input_type="jsonl",
        clear_dataset_if_exists=clear_dataset_if_exists,
    )

    # Run clustering, prepare labels, and articles
    cl_handler.run(
        generate_labels=True, generate_articles=True, check_spelling=False
    )

    sds, clusters = cl_handler.to_db_objects(
        store_to_db=True, day_to_summary=begin_date
    )

    print("==" * 50)
    print("sds.day_to_summary", sds.day_to_summary)
    print("sds.when_generated", sds.when_generated)
    print("sds.clustering", sds.clustering)
    print("==" * 50)
    clustering = sds.clustering
    print(clustering.clustering_method)
    print(clustering.clustering_options)
    print(clustering.reducer_method)
    print(clustering.reducer_optimizer)
    print(clustering.reducer_similarity)
    print(clustering.genai_article_model)
    print(clustering.genai_labels_model)
    print(clustering.genai_article_prompt[:20])
    print(clustering.genai_labels_prompt[:20])
    print("==" * 50)
    for c in clusters:
        print("label=", c.label, "label_str=", c.label_str)
        print("is_outlier=", c.is_outlier)
        print("article=", c.article_text[:10] + "...")
        print("stats=", c.stats)
        print("size=", c.size)
        print("c.sample_indices=", c.sample_indices)
        print("len(news_urls)=", len(c.news_urls))
        print("len(news_metadata)=", len(c.news_metadata))
        print("c.news_urls[:2]=", c.news_urls[:2])
        print("c.news_metadata[:2]=", c.news_metadata[:2])
        print("c.sample.size=", c.sample.size)
        print("c.sample.news_urls[:2]=", c.sample.news_urls[:2])
        print("c.sample.news_metadata[:2]=", c.sample.news_metadata[:2])
        print("--" * 50)
    print("==" * 50)

    # Unlink a temporary file
    os.unlink(news_temp_file)
    if not os.path.exists(news_temp_file):
        logging.info(f"Temporary file {news_temp_file} has been deleted.")
    else:
        logging.error(
            f"Temporary file {news_temp_file} has not been deleted. "
            f"File should be deleted manually or will be automatically "
            f"removed after next system restart."
        )

    SystemController.end_public_yesterday_news_generation(system_settings)


if __name__ == "__main__":
    main()
