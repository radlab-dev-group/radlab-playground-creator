import json
from typing import List


class ModelsConfigController:
    JSON_MAIN_FIELD = None
    JSON_MODEL_ENDPOINTS = "ep"
    JSON_MODEL_TO_HOSTS = "model_hosts"

    PUBLIC_SSE_NEWS_DEFAULT_CONFIG = "configs/public-sse-config.json"

    def __init__(self, models_config_path: str):
        self._json_dict = json.load(open(models_config_path, "rt"))
        self._assert_config()

        self._models_in_field = self._load_config_for_model_field()
        self._m2e2hosts = self._endpoints_to_available_hosts_models()

    @property
    def config_as_dict(self):
        return self._json_dict

    def available_models(self):
        return self._m2e2hosts.keys()

    def _assert_config(self):
        assert self.JSON_MAIN_FIELD is not None
        assert self.JSON_MODEL_TO_HOSTS is not None
        assert self.JSON_MODEL_ENDPOINTS is not None

    def _load_config_for_model_field(self):
        model_field_config = self._json_dict[self.JSON_MAIN_FIELD]
        return model_field_config

    def _endpoints_to_available_hosts_models(self):
        """
        Returns mapping of endpoints names to available ep-urls for each model.
        As the result the dictionary is returned. The key in the dictionary
        is the name of the model, and as the value is the dictionary of
        model configuration. Single model configuration contains mapping of
        endpoint name to all possible full-endpoint urls, connected to each
        available host for this model.
        :return:
        """
        m2ep2h = {}
        for model_name, model_config in self._models_in_field.items():
            m2ep2h[model_name] = {}
            model_hosts = list(set(model_config[self.JSON_MODEL_TO_HOSTS]))
            for ep_name, ep_url in model_config[self.JSON_MODEL_ENDPOINTS].items():
                m2ep2h[model_name][ep_name] = [
                    f"{h.strip('/')}/{ep_url.strip('/')}" for h in model_hosts
                ]
        return m2ep2h

    @staticmethod
    def _get_ep_host(available_hosts: List[str]):
        return available_hosts[0]
