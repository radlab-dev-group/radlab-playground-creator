import os
import django
import argparse

from tqdm import tqdm

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from creator.models import NewsMainPage
from creator.controllers.news import NewsControllerSimple


def _batch_list(lst, batch_size):
    """
    Yield successive batches from ``lst`` of size ``batch_size``.
    """
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Delete html_content in batches.")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=100,
        help="Size of each batch when updating records (default: 100).",
    )
    return parser.parse_args(argv or [])


def main(argv=None):
    args = _parse_args(argv)
    batch_size = args.batch_size

    for mp in NewsMainPage.objects.filter(index_page=True):
        print(f"Retrieving get_main_page_contents for {mp.main_url}")
        mp_contents_pk_list = NewsControllerSimple.get_main_page_contents_pk_list(
            main_page=mp
        )
        with tqdm(
            total=len(mp_contents_pk_list),
            desc=f"Removing MainNewsPageContent html_content for {mp.main_url}",
        ) as pbar:
            for batch_ids in _batch_list(mp_contents_pk_list, batch_size):
                objects_to_update = []
                model_cls = None
                for mpc_id in batch_ids:
                    db_obj = NewsControllerSimple.get_main_page_by_id(mpc_id)
                    if db_obj and len(db_obj.html_content.strip()):
                        db_obj.html_content = ""
                        objects_to_update.append(db_obj)
                        if model_cls is None:
                            model_cls = db_obj.__class__
                if objects_to_update and model_cls:
                    model_cls.objects.bulk_update(
                        objects_to_update, ["html_content"]
                    )
                pbar.update(len(batch_ids))

        print(f"Retrieving get_main_page_subpages for {mp.main_url}")
        mp_subpages_pk_list = NewsControllerSimple.get_main_page_subpages_pk_list(
            main_page=mp
        )
        with tqdm(
            total=len(mp_subpages_pk_list),
            desc=f"Removing NewsSubPage page_content_html for {mp.main_url}",
        ) as pbar:
            for batch_ids in _batch_list(mp_subpages_pk_list, batch_size):
                objects_to_update = []
                model_cls = None
                for mpc_id in batch_ids:
                    db_obj = NewsControllerSimple.get_main_page_subpage_by_id(mpc_id)
                    if db_obj and len(db_obj.page_content_html.strip()):
                        db_obj.page_content_html = ""
                        objects_to_update.append(db_obj)
                        if model_cls is None:
                            model_cls = db_obj.__class__
                if objects_to_update and model_cls:
                    model_cls.objects.bulk_update(
                        objects_to_update, ["page_content_html"]
                    )
                pbar.update(len(batch_ids))


if __name__ == "__main__":
    main()
