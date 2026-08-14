import os
import django
import logging

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from general.constants import DEFAULT_MODELS_CONFIG

from system.controllers import SystemController
from creator.controllers.polarity import PolarityController


def main(argv=None):
    system_settings = SystemController.get_system_settings()
    if system_settings.doing_news_polarity_3c_check:
        return

    SystemController.begin_public_news_polarity_check(system_settings)

    try:
        polarity_controller = PolarityController(
            models_config_path=DEFAULT_MODELS_CONFIG,
            add_to_db=True,
        )

        news_without_polarity = polarity_controller.get_news_without_polarity_3c()
        for news in news_without_polarity:
            res = polarity_controller.check_3c_polarity(news_list=[news])
            polarity_str = res[0].get("label")
            logging.info(f"Checked 3c polarity for news {news.pk} => {polarity_str}")
    except Exception as e:
        logging.error(e)

    SystemController.end_public_news_polarity_check(system_settings)


if __name__ == "__main__":
    main()
