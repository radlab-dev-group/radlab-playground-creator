import os
import django
import logging

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from general.constants import DEFAULT_SYSTEM_SETTINGS_ID

from system.controllers import SystemController
from creator.controllers.admin import AdminController


def main(argv=None):
    system_settings = SystemController.get_system_settings()
    if system_settings.doing_news_publ_stats:
        return

    SystemController.begin_public_make_news_stats(system_settings)
    logging.info(f"Begin public news statistics generation")

    try:
        admin_controller = AdminController()
        admin_controller.generate_news_statistics(
            make_as_last_system_stats=True, settings_id=DEFAULT_SYSTEM_SETTINGS_ID
        )
    except Exception as e:
        logging.error(e)

    logging.info(f"Public news statistics generation has ended")
    SystemController.end_public_make_news_stats(system_settings)


if __name__ == "__main__":
    main()
