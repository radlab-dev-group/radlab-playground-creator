import os
import django
import datetime
import tqdm

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from creator.models import GeneratedNews, NewsSubPage


all_generated_news = GeneratedNews.objects.all()
time_delta = datetime.timedelta(minutes=30)

with tqdm.tqdm(
    total=len(all_generated_news), desc="Updating crawling dates"
) as pbar:
    for gn in GeneratedNews.objects.all():
        if gn.when_generated < gn.news_sub_page.when_crawled:
            new_crawling_date = gn.when_generated - time_delta
            # print(
            #     "wygeneowano",
            #     gn.when_generated,
            #     "zmieniam stronę",
            #     gn.news_sub_page.pk,
            #     "z",
            #     gn.news_sub_page.when_crawled,
            #     "na",
            #     new_crawling_date
            # )
            NewsSubPage.objects.filter(pk=gn.news_sub_page.pk).update(
                when_crawled=new_crawling_date
            )
        pbar.update(1)
