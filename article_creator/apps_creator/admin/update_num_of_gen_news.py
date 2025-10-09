import os
import tqdm
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from creator.models import GeneratedNews, NewsSubPage

all_generated_news = GeneratedNews.objects.all().values_list(
    "news_sub_page__pk", flat=True
)


with tqdm.tqdm(
    total=len(all_generated_news), desc="Preparing subpage generated news"
) as pbar:
    s_id_to_gen_news = {}
    for news_sub_page_pk in all_generated_news:
        if news_sub_page_pk not in s_id_to_gen_news:
            s_id_to_gen_news[news_sub_page_pk] = 0
        s_id_to_gen_news[news_sub_page_pk] += 1
        pbar.update(1)


with tqdm.tqdm(
    total=len(s_id_to_gen_news), desc="Updating subpage generated news count"
) as pbar:
    for subpage_id, gen_news_count in s_id_to_gen_news.items():
        NewsSubPage.objects.filter(pk=subpage_id).update(
            num_of_generated_news=gen_news_count
        )
        print(f"{subpage_id} => {gen_news_count}")
        pbar.update(1)
