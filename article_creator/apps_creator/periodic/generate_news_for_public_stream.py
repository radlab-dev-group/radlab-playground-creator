import os
import random

import django
import logging

from sentence_transformers.cross_encoder import CrossEncoder

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from system.controllers import SystemController
from creator.controllers.news import NewsController
from general.constants import DEFAULT_MODELS_CONFIG
from apps_creator.periodic.src.utils import prepare_parser


def main(argv=None):
    parser = prepare_parser(argv)
    parser.add_argument(
        "--without-ce-sim", dest="without_ce_sim", action="store_true"
    )
    parser.add_argument(
        "--cross-encoder-model",
        dest="cross_encoder_model",
        default="radlab/polish-cross-encoder",
        help="Cross-Encoder model in case when similarity should be calculated "
        "between generated news and the original article.",
    )
    args = parser.parse_args()

    system_settings = SystemController.get_system_settings()
    if system_settings.doing_news_summarization:
        return

    SystemController.begin_public_news_generation(system_settings)

    try:
        ce_sim_model = None
        news_controller = NewsController(
            add_to_db=True,
            seconds_prev_check=0,
            models_config_path=DEFAULT_MODELS_CONFIG,
        )

        articles_to_summarize = (
            news_controller.public_subpages_without_summarization()
        )
        articles_to_summarize = list(articles_to_summarize)
        random.shuffle(articles_to_summarize)

        if len(articles_to_summarize) and not args.without_ce_sim:
            logging.info(f"Loading CE model {args.cross_encoder_model}...")

            ce_device = os.environ.get("CROSS_ENCODER_DEVICE", "auto")
            logging.info(f"CrossEncoder device: {ce_device}")

            ce_sim_model = CrossEncoder(args.cross_encoder_model, device=ce_device)
            logging.info(
                f"Model {args.cross_encoder_model} is loaded, "
                f"starting news generation"
            )

        all_generated_news = []
        art_to_sum_count = len(articles_to_summarize)
        for news_num, news_sub_page in enumerate(articles_to_summarize):
            logging.info(
                f"[{news_num}/{art_to_sum_count}] "
                f"Generating news for {news_sub_page.news_url}"
            )

            generated_news = news_controller.generate_news(
                news_sub_page=news_sub_page, cross_encoder_sim_model=ce_sim_model
            )
            if generated_news is None:
                logging.warning(
                    f"Problem occurred while generating news for "
                    f"{news_sub_page.news_url} "
                )
                continue

            all_generated_news.append(generated_news)

        logging.info(f"Generated {len(all_generated_news)} news")
    except Exception as e:
        logging.error(e)

    SystemController.end_public_news_generation(system_settings)


if __name__ == "__main__":
    main()
