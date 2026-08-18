import logging
import queue
import threading
from typing import List, Dict

from django import db
from llm_router_lib.client import LLMRouterClient

from creator.models import NewsSubPage, GeneratedNews
from system.models import PublicSSESettings

from general.api_utils import BasePublicApiInterface
from general.controllers.models import ModelsConfigController


class PolarityController(ModelsConfigController):
    JSON_MAIN_FIELD = "emotion_polarity"
    JSON_MAIN_FIELD_C3_POLARITY = "emotion_polarity_3c"
    POLARITY_CHECK_3C_EP = "check_3c_polarity"

    LLM_ROUTER_HOST = "llm_router_host"
    LLM_ROUTER_TOKEN = "llm_router_token"
    LLM_ROUTER_MODEL = "llm_router_model"
    LLM_ROUTER_TIMEOUT = "llm_router_timeout"

    API_HEADER = {"Content-Type": "application/json; charset=utf-8"}

    def __init__(self, models_config_path: str, add_to_db: bool = True):
        super().__init__(models_config_path)
        self._last_response = None
        self.data = None
        self.add_to_db = add_to_db

    @property
    def last_response(self):
        return self._last_response

    def __llm_router_client(self) -> LLMRouterClient:
        """
        Creates and returns an instance of LLMRouterClient configured for the
        polarity routing service.

        Returns:
            LLMRouterClient: An instance of the LLMRouterClient, initialized with
            API endpoint, authentication token, default model, and timeout.
        """
        return LLMRouterClient(
            api=self._models_in_field.get(self.LLM_ROUTER_HOST),
            token=self._models_in_field.get(self.LLM_ROUTER_TOKEN),
            default_model=self._models_in_field.get(self.LLM_ROUTER_MODEL),
            timeout=self._models_in_field.get(self.LLM_ROUTER_TIMEOUT),
        )

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

    def check_polarity_3c(self, news_list: List[GeneratedNews]) -> List[dict] | None:
        """
        Check 3-class polarity using LLM Router client
        :param news_list: List of GeneratedNews objects
        :return: List of dicts with polarity results or None on error
        """
        logging.info("Checking 3c polarity")
        if not news_list:
            return []

        texts_to_check = [n.generated_text for n in news_list]

        with self.__llm_router_client() as llm_router:
            response = llm_router.polarity_3c(texts=texts_to_check)

        if not response or "response" not in response:
            self._last_response = response
            return None

        news_polarity_response = response["response"]
        self._last_response = response

        if self.add_to_db:
            for idx, news in enumerate(news_list):
                if idx < len(news_polarity_response):
                    item = news_polarity_response[idx]
                    text_3c = item.get("original", item.get("text", None))
                    assert news.generated_text == text_3c
                    label_3c = item.get("polarity", item.get("label", None))
                    GeneratedNews.objects.filter(pk=news.pk).update(
                        polarity_3c=label_3c
                    )

        return news_polarity_response

    def check_3c_polarity(self, news_list: List[GeneratedNews]) -> List[dict] | None:
        """
        Backwards-compatible wrapper for check_polarity_3c
        """
        return self.check_polarity_3c(news_list=news_list)

    def check_polarity_3c_local(
        self, news_list: List[GeneratedNews]
    ) -> List[dict] | None:
        """
        Check 3-class polarity using local model endpoint (previous implementation)
        :param news_list: List of GeneratedNews objects
        :return: List of dicts with polarity results or None on error
        """
        logging.info("Checking 3c polarity (local)")
        if not news_list:
            return []

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

        if not response or "response" not in response:
            self._last_response = response
            return None

        news_polarity_response = response["response"]
        self._last_response = response

        if self.add_to_db:
            for idx, news in enumerate(news_list):
                if idx < len(news_polarity_response):
                    item = news_polarity_response[idx]
                    text_3c = item.get("text", item.get("original", None))
                    assert news.generated_text == text_3c
                    label_3c = item.get("label", item.get("polarity", None))
                    GeneratedNews.objects.filter(pk=news.pk).update(
                        polarity_3c=label_3c
                    )

        return news_polarity_response

    def check_polarity_3c_parallel(
        self, news_list: List[GeneratedNews], num_workers: int = 1
    ) -> List[dict]:
        """
        Check 3-class polarity in parallel using worker threads
        :param news_list: List of GeneratedNews objects
        :param num_workers: Number of parallel worker threads
        :return: List of all checked polarity responses
        """
        if not news_list:
            return []

        all_news_count = len(news_list)
        num_workers = max(1, num_workers)
        tasks_queue: queue.Queue = queue.Queue()
        results: List[dict] = []
        results_lock = threading.Lock()

        def worker():
            while True:
                item = tasks_queue.get()
                if item is None:
                    tasks_queue.task_done()
                    break

                news_num, news = item
                try:
                    logging.info(
                        f"[*] Checking 3c polarity {news_num}/{all_news_count} news_id={news.pk}"
                    )
                    res = self.check_polarity_3c(news_list=[news])
                    polarity_str = (
                        res[0].get("polarity", res[0].get("label")) if res else None
                    )
                    logging.info(
                        f"Checked 3c polarity for news {news.pk} => {polarity_str}"
                    )
                    if res:
                        with results_lock:
                            results.extend(res)
                except Exception as e:
                    logging.error(
                        f"Error while checking polarity for news {news.pk}: {e}"
                    )
                finally:
                    tasks_queue.task_done()
                    db.close_old_connections()

        threads = []
        for _ in range(num_workers):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        for news_num, news in enumerate(news_list):
            tasks_queue.put((news_num, news))

        for _ in range(num_workers):
            tasks_queue.put(None)

        for t in threads:
            t.join()

        return results
