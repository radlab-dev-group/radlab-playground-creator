import json
import logging
from typing import List, Dict

from django.db.models import QuerySet

from creator.models import NewsSubPage, GeneratedNews
from system.models import PublicSSESettings

from general.api_utils import BasePublicApiInterface
from general.controllers.models import ModelsConfigController


class PublicSSEController(ModelsConfigController):
    OVERLAP_TOKENS = 20
    MAX_TOKENS_IN_CHUNK = 200

    DEFAULT_NEWS_LANGUAGE = "pl"

    JSON_MAIN_SSE_CONFIG_FIELD = "sse_engine"
    JSON_MAIN_FIELD = "news_stream_sse_api_config"
    JSON_FIELD_COLLECTION = "news_stream_sse_collection_config"

    SSE_LOGIN_PUBLIC_EP = "login"
    SSE_ADD_INDEX_TEXTS_EP = "add_and_index_texts"
    SSE_NEW_PUBLIC_COLLECTION_EP = "new_collection"
    SSE_SEARCH_NEWS = "search_news"

    API_HEADER = {"Content-Type": "application/json; charset=utf-8"}

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self._last_response = None
        self.data = None

    @property
    def last_response(self):
        return self._last_response

    @property
    def collection_name_from_config(self):
        return self.config_as_dict[self.JSON_FIELD_COLLECTION][
            "news_collection_name"
        ]

    def add_sse_collection_from_config(self) -> int | None:
        """
        Add collection defined in config file
        :return:
        """

        collection_cfg = self._json_dict[self.JSON_FIELD_COLLECTION]
        self.data = {
            "collection_name": collection_cfg["news_collection_name"],
            "collection_description": collection_cfg["news_collection_description"],
            "collection_display_name": collection_cfg[
                "news_collection_display_name"
            ],
            "embedder_index_type": collection_cfg["news_embedder_index_type"],
            "model_embedder": collection_cfg["news_embedder_model_name"],
            "model_embedder_vector_size": collection_cfg["news_embedder_model_size"],
            "model_reranker": collection_cfg["news_reranker_model_name"],
        }

        return self.call_api_add_sse_collection_from_data_dict(self.data)

    def add_sse_collection_from_sse_public_settings(
        self, settings: PublicSSESettings
    ) -> int | None:
        """
        Add collection defined in config file
        :return:
        """
        self.data = {
            "collection_name": settings.news_collection_name,
            "collection_description": settings.news_collection_description,
            "collection_display_name": settings.news_collection_display_name,
            "embedder_index_type": settings.news_embedder_index_type,
            "model_embedder": settings.news_embedder_model_name,
            "model_embedder_vector_size": settings.news_embedder_model_size,
            "model_reranker": settings.news_reranker_model_name,
        }

        return self.call_api_add_sse_collection_from_data_dict(self.data)

    def call_api_add_sse_collection_from_data_dict(self, data: Dict) -> int | None:
        logging.info(f"Adding collection {data['collection_name']}")

        ep_url = self._get_ep_host(
            self._m2e2hosts[self.JSON_MAIN_SSE_CONFIG_FIELD][
                self.SSE_NEW_PUBLIC_COLLECTION_EP
            ]
        )
        login_ep_url = self._get_ep_host(
            self._m2e2hosts[self.JSON_MAIN_SSE_CONFIG_FIELD][
                self.SSE_LOGIN_PUBLIC_EP
            ]
        )

        response = BasePublicApiInterface.general_call_post(
            host_url=None,
            endpoint=ep_url,
            data=None,
            json_data=data,
            headers=self.API_HEADER,
            login_url=login_ep_url,
        )

        if (
            "body" not in response
            or "status" not in response
            or response["status"] is False
        ):
            self._last_response = response
            return None
        return response["body"]["collection_id"]

    def add_and_index_news_to_sse(self):
        assert self.data is not None

        collection_name = self.data["collection_name"]
        ep_url = self._get_ep_host(
            self._m2e2hosts[self.JSON_MAIN_SSE_CONFIG_FIELD][
                self.SSE_ADD_INDEX_TEXTS_EP
            ]
        )
        login_ep_url = self._get_ep_host(
            self._m2e2hosts[self.JSON_MAIN_SSE_CONFIG_FIELD][
                self.SSE_LOGIN_PUBLIC_EP
            ]
        )

        gen_news_to_index_in_sse = self._load_news_to_index_in_sse()
        all_news_count = len(gen_news_to_index_in_sse)
        for news_num, gen_news in enumerate(gen_news_to_index_in_sse):
            logging.info(
                f"[*] Indexing news {news_num}/{all_news_count} news_id={gen_news.pk}"
            )

            news_to_index = [gen_news]
            response_body = self._call_sse_api_to_index_texts(
                collection_name=collection_name,
                generated_news_to_index=news_to_index,
                ep_url=ep_url,
                login_ep_url=login_ep_url,
            )

            if response_body is not None:
                NewsSubPage.objects.filter(pk=gen_news.news_sub_page.pk).update(
                    is_indexed_in_sse=True
                )

                indexed_chunks = response_body.get("indexed_chunks", 0)
                indexed_documents = response_body.get("indexed_documents", 0)
                logging.info(
                    f"    -> indexed documents={indexed_documents} "
                    f"chunks={indexed_chunks}"
                )

    def search_news(
        self, text_to_search: str, num_of_results: int, relative_paths: list
    ) -> dict:
        ep_url = self._get_ep_host(
            self._m2e2hosts[self.JSON_MAIN_SSE_CONFIG_FIELD][self.SSE_SEARCH_NEWS]
        )
        login_ep_url = self._get_ep_host(
            self._m2e2hosts[self.JSON_MAIN_SSE_CONFIG_FIELD][
                self.SSE_LOGIN_PUBLIC_EP
            ]
        )

        data_api_call = {
            "query_str": text_to_search,
            "collection_name": self.collection_name_from_config,
            "options": json.dumps(
                {
                    "relative_paths": relative_paths,
                    "rerank_results": False,
                    "return_with_factored_fields": False,
                    "max_results": num_of_results,
                }
            ),
            "ignore_question_lang_detect": True,
        }
        api_response = BasePublicApiInterface.general_call_post(
            host_url=None,
            endpoint=ep_url,
            data=None,
            json_data=data_api_call,
            headers=self.API_HEADER,
            login_url=login_ep_url,
        )
        if (
            "body" not in api_response
            or "status" not in api_response
            or api_response["status"] is False
        ):
            self._last_response = api_response
            return {}
        return api_response["body"]

    def _call_sse_api_to_index_texts(
        self,
        collection_name: str,
        generated_news_to_index: List[GeneratedNews],
        ep_url: str,
        login_ep_url: str,
    ):
        documents_as_dicts = self._prepare_news_as_doc_dict(
            generated_news_to_index=generated_news_to_index
        )
        logging.info(f" -> calling api to index {len(documents_as_dicts)} text(s)")

        data = {
            "texts[]": documents_as_dicts,
            "collection_name": collection_name,
            "indexing_options": {
                "clear_text": False,
                "use_text_denoiser": False,
                "check_text_lang": True,
                "prepare_proper_pages": False,
                "merge_document_pages": False,
                "max_tokens_in_chunk": self.MAX_TOKENS_IN_CHUNK,
                "number_of_overlap_tokens": self.OVERLAP_TOKENS,
            },
        }

        response = BasePublicApiInterface.general_call_post(
            host_url=None,
            endpoint=ep_url,
            data=None,
            json_data=data,
            headers=self.API_HEADER,
            login_url=login_ep_url,
        )

        if (
            "body" not in response
            or "status" not in response
            or response["status"] is False
        ):
            self._last_response = response
            return None
        return response["body"]

    @staticmethod
    def _prepare_news_as_doc_dict(
        generated_news_to_index: List[GeneratedNews],
    ) -> List[Dict]:
        news_as_dict = []

        for generated_mews in generated_news_to_index:
            news_sub_page = generated_mews.news_sub_page
            text_to_index = generated_mews.generated_text
            if text_to_index is None or not len(text_to_index):
                continue

            n_dict = {
                "filepath": news_sub_page.news_url.rstrip("/"),
                "relative_filepath": news_sub_page.news_url,
                "category": news_sub_page.main_page.category.display_name,
                "options": {},
                "pages": [
                    {
                        "text_chunk_type": "news",
                        "page_number": 1,
                        # "table_number": 0,
                        # "row_number": 0,
                        # "column_number": 0,
                        "page_content": text_to_index,
                    }
                ],
            }
            news_as_dict.append(n_dict)
        return news_as_dict

    @staticmethod
    def _load_news_to_index_in_sse() -> QuerySet[GeneratedNews]:
        return GeneratedNews.objects.filter(
            show_news=True,
            news_sub_page__is_indexed_in_sse=False,
        ).order_by("-news_sub_page__when_crawled")
