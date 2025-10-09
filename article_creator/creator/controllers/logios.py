import json

from creator.models import GeneratedNews
from general.api_utils import BasePublicApiInterface
from general.constants import DEFAULT_LOGIOS_CONFIG


class LogiosController:
    MAIN_API_HOST_FIELD = "api_host"
    MAIN_API_EP_URL_FIELD = "ep_url"
    MAIN_API_EP_DEF_PARAMS_FIELD = "default_params"

    MAIN_REDAKTOR_JSON_FIELD = "logios_redaktor"
    REDAKTOR_CHECK_PLI_EP = "check_pli"

    LOGIOS_PLI_MEASURE_NAME = "PLI_Logios"

    def __init__(
        self, config_path: str = DEFAULT_LOGIOS_CONFIG, add_to_db: bool = True
    ):
        self._add_to_db = add_to_db
        self._config_path = config_path

        self._whole_config = None
        self._redaktor_config = None
        self._redaktor_endpoints = {}

        if len(config_path):
            self.__load_config_from_path()

    def get_check_pli_ep_url(self):
        if not self.__verify_redaktor_config():
            return None
        return self._redaktor_endpoints[self.REDAKTOR_CHECK_PLI_EP][
            self.MAIN_API_EP_URL_FIELD
        ]

    def get_check_pli_ep_default_params(self):
        if not self.__verify_redaktor_config():
            return None
        return self._redaktor_endpoints[self.REDAKTOR_CHECK_PLI_EP][
            self.MAIN_API_EP_DEF_PARAMS_FIELD
        ]

    def check_news_pli(self, generated_news: GeneratedNews) -> float | None:
        if generated_news is None:
            return None

        ep_url = self.get_check_pli_ep_url()
        data = self.get_check_pli_ep_default_params()
        data["text_str"] = generated_news.news_sub_page.page_content_txt
        response = BasePublicApiInterface.general_call_post(
            host_url=None,
            endpoint=ep_url,
            data=data,
        )
        if "status" not in response or not response["status"]:
            return None

        pli_value = self.__get_pli_value_from_response(response)
        if pli_value is None:
            return None

        if self._add_to_db:
            GeneratedNews.objects.filter(pk=generated_news.pk).update(
                pli_value=pli_value
            )
        return pli_value

    @staticmethod
    def get_news_without_pli():
        return list(
            GeneratedNews.objects.filter(pli_value=None).order_by("-when_generated")
        )

    def __get_pli_value_from_response(self, api_response) -> float | None:
        all_measures = api_response.get("calculated_measures", {})
        for measure in all_measures:
            if measure["measure"]["name"] == self.LOGIOS_PLI_MEASURE_NAME:
                return measure["value"]
        return None

    def __verify_redaktor_config(self):
        if (
            self._redaktor_endpoints is None
            or self.MAIN_API_EP_URL_FIELD
            not in self._redaktor_endpoints[self.REDAKTOR_CHECK_PLI_EP]
            or self.MAIN_API_EP_DEF_PARAMS_FIELD
            not in self._redaktor_endpoints[self.REDAKTOR_CHECK_PLI_EP]
        ):
            return False
        return True

    def __load_config_from_path(self):
        with open(self._config_path, "r") as config_file:
            self._whole_config = json.load(config_file)
            self._redaktor_config = self._whole_config.get(
                self.MAIN_REDAKTOR_JSON_FIELD, None
            )
            if self._redaktor_config is not None:
                api_host = self._redaktor_config[self.MAIN_API_HOST_FIELD].rstrip(
                    "/"
                )
                full_pli_ep = self._redaktor_config["ep"][self.REDAKTOR_CHECK_PLI_EP]
                ep_url = full_pli_ep[self.MAIN_API_EP_URL_FIELD].rstrip("/")
                ep_def_params = full_pli_ep[self.MAIN_API_EP_DEF_PARAMS_FIELD]
                full_ep_url = f"{api_host}/{ep_url}"
                self._redaktor_endpoints[self.REDAKTOR_CHECK_PLI_EP] = {
                    self.MAIN_API_EP_URL_FIELD: full_ep_url,
                    self.MAIN_API_EP_DEF_PARAMS_FIELD: ep_def_params,
                }
