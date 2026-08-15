import os
import django
import logging

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from general.constants import DEFAULT_LOGIOS_CONFIG

from system.controllers import SystemController
from creator.controllers.logios import LogiosController


def main(argv=None):
    system_settings = SystemController.get_system_settings()
    if system_settings.doing_news_pli_check:
        return

    SystemController.begin_public_news_pli_check(system_settings)

    try:
        logios_controller = LogiosController(
            config_path=DEFAULT_LOGIOS_CONFIG,
            add_to_db=True,
        )

        news_without_pli = logios_controller.get_news_without_pli()
        for news in news_without_pli:
            pli_val = logios_controller.check_news_pli(generated_news=news)
            logging.info(f"Checked PLI value for news {news.pk} = {pli_val}")
    except Exception as e:
        logging.error(e)

    SystemController.end_public_news_pli_check(system_settings)


if __name__ == "__main__":
    main()
