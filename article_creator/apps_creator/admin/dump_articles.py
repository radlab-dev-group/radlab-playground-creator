import json
import os
import tqdm
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from creator.models import GeneratedNews, NewsSubPage

all_generated_news = NewsSubPage.objects.all().values_list(
    "page_content_txt", flat=True
)

out_filename = "plg-articles-20250407.jsonl"

with open(out_filename, "w") as f:
    with tqdm.tqdm(
        total=len(all_generated_news), desc="Dumping subpage generated news"
    ) as pbar:
        for page_content_txt in all_generated_news:
            c_str = page_content_txt.strip()

            if len(c_str):
                out_data = {"text": c_str}
                f.write(json.dumps(out_data, ensure_ascii=False) + "\n")

            pbar.update(1)
