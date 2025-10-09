import os
import random

import django
import logging

from sentence_transformers.cross_encoder import CrossEncoder

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from general.constants import DEFAULT_MODELS_CONFIG

from creator.controllers.news import NewsController
from system.controllers import SystemController

CROSS_ENCODER_MODEL = "radlab/polish-cross-encoder"


def main(argv=None):
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
        if len(articles_to_summarize):
            logging.info(f"Loading CE model {CROSS_ENCODER_MODEL}...")
            ce_sim_model = CrossEncoder(CROSS_ENCODER_MODEL)
            logging.info(
                f"Model {CROSS_ENCODER_MODEL} is loaded, starting news generation"
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
                continue
            all_generated_news.append(generated_news)

        logging.info(f"Generated {len(all_generated_news)} news")
    except Exception as e:
        logging.error(e)

    SystemController.end_public_news_generation(system_settings)


if __name__ == "__main__":
    main()
