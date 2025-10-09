import logging
from typing import List, Dict

from creator.models import NewsSubPage, GeneratedNews
from system.models import PublicSSESettings

from general.api_utils import BasePublicApiInterface
from general.controllers.models import ModelsConfigController


class PolarityController(ModelsConfigController):
    JSON_MAIN_FIELD = "emotion_polarity"
    JSON_MAIN_FIELD_C3_POLARITY = "emotion_polarity_3c"
    POLARITY_CHECK_3C_EP = "check_3c_polarity"

    API_HEADER = {"Content-Type": "application/json; charset=utf-8"}

    def __init__(self, models_config_path: str, add_to_db: bool = True):
        super().__init__(models_config_path)
        self._last_response = None
        self.data = None
        self.add_to_db = add_to_db

    @property
    def last_response(self):
        return self._last_response

    @staticmethod
    def get_news_without_polarity_3c():
        return list(
            GeneratedNews.objects.filter(polarity_3c=None).order_by(
                "-when_generated"
            )
        )

    @staticmethod
    def get_all_news():
        return list(GeneratedNews.objects.all().order_by("-when_generated"))

    def check_3c_polarity(self, news_list: List[GeneratedNews]) -> List[dict] | None:
        """
        Add collection defined in config file
        :return:
        """
        logging.info(f"Checking 3c polarity")
        ep_url = self._get_ep_host(
            self._m2e2hosts[self.JSON_MAIN_FIELD_C3_POLARITY][
                self.POLARITY_CHECK_3C_EP
            ]
        )

        texts_to_check = [n.generated_text for n in news_list]
        data = {"texts": texts_to_check}

        response = BasePublicApiInterface.general_call_post(
            host_url=None,
            endpoint=ep_url,
            data=None,
            json_data=data,
            headers=self.API_HEADER,
            login_url=None,
        )

        if "response" not in response:
            self._last_response = response
            return None
        news_polarity_response = response["response"]
        if self.add_to_db:
            for idx, news in enumerate(news_list):
                text_3c = news_polarity_response[idx].get("text", None)
                assert news.generated_text == text_3c
                label_3c = news_polarity_response[idx].get("label", None)
                GeneratedNews.objects.filter(pk=news.pk).update(polarity_3c=label_3c)
        return news_polarity_response
