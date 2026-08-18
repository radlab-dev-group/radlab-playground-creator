import os
import django
import logging

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from general.constants import DEFAULT_MODELS_CONFIG
from apps_creator.periodic.src.utils import prepare_parser

from system.controllers import SystemController
from creator.controllers.polarity import PolarityController


def prepare_polarity_parser(argv=None):
    parser = prepare_parser(argv)
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=1,
        help="Number of workers for parallel news polarity checking.",
    )
    return parser


def main(argv=None):
    args = prepare_polarity_parser(argv).parse_args(argv)

    system_settings = SystemController.get_system_settings()
    if system_settings.doing_news_polarity_3c_check:
        return

    SystemController.begin_public_news_polarity_check(system_settings)

    try:
        polarity_controller = PolarityController(
            models_config_path=args.json_config or DEFAULT_MODELS_CONFIG,
            add_to_db=True,
        )

        news_without_polarity = polarity_controller.get_news_without_polarity_3c()
        polarity_controller.check_polarity_3c_parallel(
            news_list=news_without_polarity,
            num_workers=args.num_workers,
        )
    except Exception as e:
        logging.error(e)

    SystemController.end_public_news_polarity_check(system_settings)


if __name__ == "__main__":
    main()
