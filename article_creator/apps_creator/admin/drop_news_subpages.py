import os
import django

from tqdm import tqdm


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from creator.models import GeneratedNews
from creator.controllers.news import NewsControllerSimple


def drop_objects(obj_list, tqdm_descr: str):
    if obj_list is None or not len(obj_list):
        return
    with tqdm(total=len(obj_list), desc=tqdm_descr) as pbar:
        for obj in obj_list:
            obj.delete()
            pbar.update(1)


def main(argv=None):
    main_url = None

    if main_url is None:
        raise Exception("No main url provided")

    main_pages = NewsControllerSimple.get_main_page(main_url=main_url)
    if not len(main_pages):
        print(f"No news pages found for {main_url}")
        return

    print(f"Fetching sub-pages and contents for {len(main_pages)} instances")
    for mp in main_pages:
        mp_contents = NewsControllerSimple.get_main_page_contents(main_page=mp)
        drop_objects(
            obj_list=mp_contents,
            tqdm_descr=f"Deleting content for main page={mp.pk}",
        )

        mp_subpages = NewsControllerSimple.get_main_page_subpages(main_page=mp)
        for sub in mp_subpages:
            for gn in GeneratedNews.objects.filter(news_sub_page=sub):
                gn.delete()

        drop_objects(
            obj_list=mp_subpages,
            tqdm_descr=f"Deleting subpages for main page={mp.pk}",
        )


if __name__ == "__main__":
    main()
