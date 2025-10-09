import os
import json
import django
import logging

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from system.controllers import SystemController
from creator.controllers.sse_engine_public import PublicSSEController
from apps_creator.periodic.src.utils import prepare_parser


def main(argv=None):
    args = prepare_parser(argv).parse_args(argv)

    system_settings = SystemController.get_system_settings()
    if system_settings.doing_news_semantic_indexing:
        return

    SystemController.begin_public_news_indexing_sse(system_settings)

    p_settings = system_settings.public_stream_news_sse_settings
    if p_settings is None:
        p_settings = SystemController.get_public_sse_news_stream_settings(
            system_settings=system_settings,
            sse_settings_dict=json.load(open(args.json_config, "rt"))[
                PublicSSEController.JSON_FIELD_COLLECTION
            ],
        )

    try:
        sse_controller = PublicSSEController(config_path=args.json_config)
        sse_controller.add_sse_collection_from_sse_public_settings(p_settings)
        sse_controller.add_and_index_news_to_sse()
    except Exception as e:
        logging.error(e)

    SystemController.end_public_news_indexing_sse(system_settings)


if __name__ == "__main__":
    main()
