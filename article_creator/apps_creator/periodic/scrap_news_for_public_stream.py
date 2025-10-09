import os
import django
import logging

from typing import Dict, List

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from system.controllers import SystemController
from creator.controllers.news import MainNewsController, NewsController
from apps_creator.periodic.src.utils import prepare_parser, load_json_config


def add_categories(
    categories_dict: Dict,
    main_news_controller: MainNewsController,
    debug: bool = False,
) -> Dict:
    categories = {}
    for cat_name, cat_opts in categories_dict.items():
        if debug:
            logging.info(f"Adding category {cat_name}")
        categories[cat_name] = main_news_controller.get_add_category(
            name=cat_name,
            display_name=cat_opts["display_name"],
            description=cat_opts["description"],
            order=cat_opts["order"],
        )

    return categories


def add_main_urls(
    main_urls_dict: List,
    news_controller: MainNewsController,
    categories: Dict,
    debug: bool = False,
) -> Dict:
    urls_dict = {}

    for main_url_cfg in main_urls_dict:
        main_url = main_url_cfg["main_url"]
        main_url_category = categories[main_url_cfg["category"]]

        if debug:
            logging.info(
                f"Adding main url {main_url} with "
                f"category {main_url_category.display_name}"
            )

        urls_dict[main_url] = news_controller.get_add_main_public_url(
            main_url_str=main_url,
            begin_crawling_url=main_url_cfg["begin_crawling_url"].strip(),
            category=main_url_category,
            index_page=main_url_cfg["index_page"],
            index_page_sse=main_url_cfg["index_page_sse"],
            prepare_news=main_url_cfg["prepare_news"],
            show_news=main_url_cfg["show_news"],
            min_news_depth=main_url_cfg["min_news_depth"],
            include_paths_with=main_url_cfg["include_paths_with"],
            exclude_paths_with=main_url_cfg["exclude_paths_with"],
            news_url_ends_with=main_url_cfg["news_url_ends_with"],
            use_heuristic_to_extract_news=main_url_cfg[
                "use_heuristic_to_extract_news_content"
            ],
            use_heuristic_to_search_news_content=main_url_cfg[
                "use_heuristic_to_search_news_content"
            ],
            use_heuristic_to_clear_news_content=main_url_cfg[
                "use_heuristic_to_clear_news_content"
            ],
            news_link_starts_with_main_url=main_url_cfg[
                "news_link_starts_with_main_url"
            ],
            language=main_url_cfg["language"],
            single_news_content_tag_name=main_url_cfg[
                "single_news_content_tag_name"
            ],
            single_news_content_tag_attr_name=main_url_cfg[
                "single_news_content_tag_attr_name"
            ],
            single_news_content_tag_attr_value=main_url_cfg[
                "single_news_content_tag_attr_value"
            ],
            remove_n_last_elems=main_url_cfg["remove_n_last_elems"],
        )

    return urls_dict


def main(argv=None):
    parser = prepare_parser(argv)
    parser.add_argument("--debug", dest="debug", action="store_true")
    args = parser.parse_args()

    system_settings = SystemController.get_system_settings()
    if system_settings.doing_news_indexing:
        return

    SystemController.begin_public_news_indexing(system_settings)

    try:
        main_news_controller = MainNewsController(add_to_db=True)
        news_controller = NewsController(add_to_db=True, seconds_prev_check=0)
        json_config = load_json_config(args.json_config)
        categories = add_categories(
            json_config.get("categories", {}), main_news_controller, debug=False
        )
        add_main_urls(
            json_config.get("main_urls", []), main_news_controller, categories
        )
        pages_contents = news_controller.download_news_main_pages_contents(
            user=None, debug_for_new_site=args.debug
        )

        _ = news_controller.download_news_subpages(pages_contents)
    except Exception as e:
        logging.error(e)
        raise e

    SystemController.end_public_news_indexing(system_settings)


if __name__ == "__main__":
    main()
