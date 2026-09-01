import os
import sys
import unittest
from unittest.mock import MagicMock

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")

# Local dev dependencies (clusterer / llm_router_lib) may not be installed in
# the test environment. If they are missing, make them importable from the
# sibling development checkouts (../clusterer, ../llm-router).
def _ensure_importable(module_name: str, candidate_root: str) -> None:
    try:
        __import__(module_name)
    except ModuleNotFoundError:
        if os.path.isdir(candidate_root):
            sys.path.insert(0, candidate_root)


_DEV_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 5)
)
_ensure_importable("clusterer", os.path.join(_DEV_ROOT, "clusterer"))
_ensure_importable("llm_router_lib", os.path.join(_DEV_ROOT, "llm-router"))

import django  # noqa: E402

django.setup()

from creator.controllers.clustering import Cluster  # noqa: E402


def _make_cluster() -> Cluster:
    cluster = Cluster(label=0)
    cluster.append(
        text="Pierwsza wiadomosc",
        proper_text="Pierwsza wiadomosc",
        metadata={"news_url": "http://example.com/1"},
    )
    cluster.append(
        text="Druga wiadomosc",
        proper_text="Druga wiadomosc",
        metadata={"news_url": "http://example.com/2"},
    )
    cluster.random_proper_texts = [
        "Pierwsza wiadomosc",
        "Druga wiadomosc",
    ]
    return cluster


def _label_response(label: str) -> MagicMock:
    """Mimics the new typed LLMRouterClient response for /api/generate_label."""
    resp = MagicMock()
    resp.response = label
    resp.generation_time = 1.0
    return resp


def _article_response(article_text: str) -> MagicMock:
    """Mimics the typed response for /api/generate_article_from_texts."""
    resp = MagicMock()
    resp.response = MagicMock()
    resp.response.article_text = article_text
    resp.generation_time = 1.0
    return resp


class TestClusterRouterCalls(unittest.TestCase):
    def test_generate_label_success(self):
        cluster = _make_cluster()
        llm_router = MagicMock()
        llm_router.generate_label.return_value = _label_response("Moja *kategoria*")

        cluster.generate_label_with_genai(llm_router=llm_router)

        llm_router.generate_label.assert_called_once_with(
            texts=cluster.random_proper_texts
        )
        self.assertEqual(cluster.generated_label, "Moja kategoria")

    def test_generate_label_empty_response(self):
        cluster = _make_cluster()
        llm_router = MagicMock()
        llm_router.generate_label.return_value = _label_response(None)

        cluster.generate_label_with_genai(llm_router=llm_router)

        self.assertEqual(cluster.generated_label, "")

    def test_generate_label_router_exception(self):
        cluster = _make_cluster()
        llm_router = MagicMock()
        llm_router.generate_label.side_effect = RuntimeError("router down")

        cluster.generate_label_with_genai(llm_router=llm_router)

        self.assertEqual(cluster.generated_label, "")

    def test_generate_article_success(self):
        cluster = _make_cluster()
        llm_router = MagicMock()
        llm_router.generate_article_from_texts.return_value = _article_response(
            "Dziśnie był dzień\nTreść artykułu"
        )

        cluster.generate_article(llm_router=llm_router)

        llm_router.generate_article_from_texts.assert_called_once_with(
            texts=cluster.random_proper_texts
        )
        self.assertIsNotNone(cluster.generated_article)
        self.assertTrue(cluster.generated_article.startswith("### "))
        self.assertIn("Dzisie był dzień", cluster.generated_article)
        self.assertNotIn("Dziśnie", cluster.generated_article)

    def test_generate_article_empty_article_text(self):
        cluster = _make_cluster()
        llm_router = MagicMock()
        llm_router.generate_article_from_texts.return_value = _article_response(None)

        cluster.generate_article(llm_router=llm_router)

        self.assertIsNone(cluster.generated_article)

    def test_generate_article_router_exception(self):
        cluster = _make_cluster()
        llm_router = MagicMock()
        llm_router.generate_article_from_texts.side_effect = RuntimeError(
            "router down"
        )

        cluster.generate_article(llm_router=llm_router)

        self.assertIsNone(cluster.generated_article)


if __name__ == "__main__":
    unittest.main()
