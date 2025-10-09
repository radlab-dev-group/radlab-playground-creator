import json
import logging
import tempfile
import argparse

from typing import Dict, Optional
from django.db.models import QuerySet

from creator.models import GeneratedNews


def prepare_parser(desc=""):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("-c", "--json-config", dest="json_config")
    return p


def load_json_config(json_config_path: str) -> Dict:
    return json.load(open(json_config_path, "rt"))


class InformationBrowser:
    TEXT_COLUMN_NAME = "text"
    METADATA_COLUMN_NAME = "metadata"

    @staticmethod
    def convert_news_to_store_jsonl(
        generated_news: QuerySet[GeneratedNews],
    ) -> list[dict]:
        con_news = []

        for news in generated_news:
            news = {
                InformationBrowser.TEXT_COLUMN_NAME: news.generated_text,
                InformationBrowser.METADATA_COLUMN_NAME: {
                    "language": news.language,
                    "pli_value": news.pli_value,
                    "polarity_3c": news.polarity_3c,
                    "source": news.news_sub_page.main_page.main_url,
                    "news_url": news.news_sub_page.news_url,
                    "news_pk": news.pk,
                },
            }
            con_news.append(news)
        return con_news

    @staticmethod
    def store_converted_news_to_jsonl_file(
        all_news: list[dict], out_file_path: Optional[str] = None
    ) -> str:
        if out_file_path is None:
            out_file_path = tempfile.NamedTemporaryFile().name

        logging.info(f"Storing news to file {out_file_path}")

        with open(out_file_path, "wt") as out_file:
            for news in all_news:
                n_str = json.dumps(news, ensure_ascii=False) + "\n"
                out_file.write(n_str)
        return out_file_path
