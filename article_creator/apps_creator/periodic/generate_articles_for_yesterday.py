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
from apps_creator.periodic.src.utils import InformationBrowser, parse_date


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
    p.add_argument(
        "--begin-date",
        dest="begin_date",
        help="Begin date in format YYYY-MM-DD",
        type=parse_date,
        required=False,
    )
    p.add_argument(
        "--end-date",
        dest="end_date",
        help="End date in format YYYY-MM-DD",
        type=parse_date,
        required=False,
    )

    return p


def generate_articles_for_day(
    news_controller: NewsController,
    cl_handler: ClusteringHandler,
    target_day: datetime.date,
    clear_dataset_if_exists: bool = True,
):
    end_date = target_day + datetime.timedelta(days=1)
    begin_date = target_day
    logging.info(f"Generating news for days {begin_date} - {end_date}")

    # Gen news for date range
    gen_news = news_controller.public_get_generated_news_for_date_range(
        begin_date=begin_date, end_date=end_date
    )

    # Convert news to pairs of text and metadata
    conv_news = InformationBrowser.convert_news_to_store_jsonl(
        generated_news=gen_news
    )
    if not conv_news:
        logging.info(f"No news found for date {target_day}, skipping.")
        return None

    news_temp_file = None
    try:
        # Store converted news to a temporary file
        news_temp_file = InformationBrowser.store_converted_news_to_jsonl_file(
            all_news=conv_news, out_file_path=None
        )
        if not news_temp_file or not os.path.exists(news_temp_file):
            logging.error(f"Temporary file {news_temp_file} does not exists!")
            logging.error(f"Cannot continue news generation! Check file existing.")
            return None

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

        return sds, clusters
    finally:
        # Unlink a temporary file
        if news_temp_file and os.path.exists(news_temp_file):
            os.unlink(news_temp_file)
            if not os.path.exists(news_temp_file):
                logging.info(f"Temporary file {news_temp_file} has been deleted.")
            else:
                logging.error(
                    f"Temporary file {news_temp_file} has not been deleted. "
                    f"File should be deleted manually or will be automatically "
                    f"removed after next system restart."
                )


def main(argv=None):
    args = prepare_parser().parse_args(argv)

    system_settings = SystemController.get_system_settings()
    if system_settings.doing_news_generation_for_yesterday:
        return

    SystemController.begin_public_yesterday_news_generation(system_settings)

    try:
        news_controller = NewsController(
            add_to_db=True,
            seconds_prev_check=0,
            models_config_path=None,
        )

        # Prepare clusterer
        cl_handler = ClusteringHandler(
            clustering_config_path="configs/clusterer-config.json",
            min_cluster_count=args.min_cluster_count,
            opt_cluster_count=args.opt_cluster_count,
            max_cluster_count=args.max_cluster_count,
        )

        if args.begin_date is None and args.end_date is None:
            yesterday = datetime.datetime.now().date() - datetime.timedelta(days=1)
            begin_date = yesterday
            end_date = yesterday
        elif args.begin_date is not None and args.end_date is None:
            begin_date = args.begin_date
            end_date = args.begin_date
        elif args.begin_date is None and args.end_date is not None:
            begin_date = args.end_date
            end_date = args.end_date
        else:
            begin_date = args.begin_date
            end_date = args.end_date

        current_date = begin_date
        while current_date <= end_date:
            generate_articles_for_day(
                news_controller=news_controller,
                cl_handler=cl_handler,
                target_day=current_date,
            )
            current_date += datetime.timedelta(days=1)
    except Exception as e:
        logging.error(f"Error during news generation: {e}")
    finally:
        SystemController.end_public_yesterday_news_generation(system_settings)


if __name__ == "__main__":
    main()
