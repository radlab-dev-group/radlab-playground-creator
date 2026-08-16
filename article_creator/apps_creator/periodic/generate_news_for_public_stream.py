import logging
import os
import random

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from apps_creator.periodic.src.news_generator import (
    generate_news_parallel,
    load_cross_encoder_model,
)
from apps_creator.periodic.src.utils import parse_date, prepare_parser
from creator.controllers.news import NewsController
from general.constants import DEFAULT_MODELS_CONFIG
from system.controllers import SystemController


def prepare_generation_parser(argv=None):
    parser = prepare_parser(argv)
    parser.add_argument(
        "--without-ce-sim",
        dest="without_ce_sim",
        action="store_true",
        help="Skip Cross-Encoder similarity calculation.",
    )
    parser.add_argument(
        "--cross-encoder-model",
        dest="cross_encoder_model",
        default="radlab/polish-cross-encoder",
        help="Cross-Encoder model in case when similarity should be calculated "
        "between generated news and the original article.",
    )
    parser.add_argument(
        "--begin-date",
        dest="begin_date",
        help="Begin date in format YYYY-MM-DD",
        type=parse_date,
        required=False,
    )
    parser.add_argument(
        "--end-date",
        dest="end_date",
        help="End date in format YYYY-MM-DD",
        type=parse_date,
        required=False,
    )
    parser.add_argument(
        "--dont-shuffle",
        dest="dont_shuffle",
        action="store_true",
        help="Disable shuffling of articles before generation.",
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=1,
        help="Number of workers for parallel news generation.",
    )
    return parser


def main(argv=None):
    parser = prepare_generation_parser(argv)
    args = parser.parse_args(argv)

    system_settings = SystemController.get_system_settings()
    if system_settings.doing_news_summarization:
        return

    SystemController.begin_public_news_generation(system_settings)

    try:
        news_controller = NewsController(
            add_to_db=True,
            seconds_prev_check=0,
            models_config_path=DEFAULT_MODELS_CONFIG,
        )

        articles_to_summarize = list(
            news_controller.public_subpages_without_summarization(
                begin_date=args.begin_date,
                end_date=args.end_date,
            )
        )
        logging.info(f"Number of articles: {len(articles_to_summarize)}")

        if not args.dont_shuffle:
            random.shuffle(articles_to_summarize)

        ce_sim_model = None
        if articles_to_summarize and not args.without_ce_sim:
            ce_sim_model = load_cross_encoder_model(
                model_name=args.cross_encoder_model
            )
            logging.info("Starting news generation...")

        all_generated_news = generate_news_parallel(
            news_controller=news_controller,
            articles=articles_to_summarize,
            cross_encoder_model=ce_sim_model,
            num_workers=args.num_workers,
        )

        logging.info(f"Generated {len(all_generated_news)} news")
    except Exception as e:
        logging.error(f"Error during news generation: {e}")
    finally:
        SystemController.end_public_news_generation(system_settings)


if __name__ == "__main__":
    main()
