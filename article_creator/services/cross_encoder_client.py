import logging
import os
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class CrossEncoderApiClient:
    """
    Client for querying the Cross-Encoder Flask API service.
    Configured with connection pooling for concurrent requests from multiple threads.
    """

    DEFAULT_API_URL = "http://127.0.0.1:8085"

    def __init__(
        self,
        api_url: Optional[str] = None,
        timeout: float = 30.0,
        pool_connections: int = 50,
        pool_maxsize: int = 50,
        max_retries: int = 2,
    ):
        raw_url = api_url or os.environ.get(
            "CROSS_ENCODER_API_URL", self.DEFAULT_API_URL
        )
        self.base_url = raw_url.rstrip("/")
        if self.base_url.endswith("/calculate_similarity") or self.base_url.endswith(
            "/api/calculate_similarity"
        ):
            self.endpoint_url = self.base_url
        else:
            self.endpoint_url = f"{self.base_url}/calculate_similarity"

        self.timeout = timeout
        self.session = requests.Session()

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
            allowed_methods=["POST", "GET"],
        )
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=retry_strategy,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def calculate_similarity(
        self,
        gen_text: Optional[str],
        orig_text: Optional[str],
        ce_model_max_tokens: int = 512,
        special_tokens: int = 2,
    ) -> Optional[float]:
        """
        Send similarity calculation request to Cross-Encoder API.

        :param gen_text: Generated article text
        :param orig_text: Original article text
        :param ce_model_max_tokens: Max tokens threshold for the model
        :param special_tokens: Number of special tokens reserved
        :return: Similarity score as float or None
        """
        if (
            not gen_text
            or not orig_text
            or not gen_text.strip()
            or not orig_text.strip()
        ):
            return None

        payload: Dict[str, Any] = {
            "gen_text": gen_text.strip(),
            "orig_text": orig_text.strip(),
            "ce_model_max_tokens": ce_model_max_tokens,
            "special_tokens": special_tokens,
        }

        try:
            response = self.session.post(
                self.endpoint_url,
                json=payload,
                headers={"Content-Type": "application/json; charset=utf-8"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            res_json = response.json()
            similarity = res_json.get("similarity")
            if similarity is not None:
                return float(similarity)
            return None
        except Exception as e:
            logger.error(
                f"Failed to calculate similarity via Cross-Encoder API ({self.endpoint_url}): {e}"
            )
            return None

    def check_health(self) -> bool:
        """
        Check if the Cross-Encoder API is healthy and reachable.
        """
        try:
            health_url = f"{self.base_url}/health"
            res = self.session.get(health_url, timeout=5.0)
            return res.status_code == 200
        except Exception:
            return False

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
