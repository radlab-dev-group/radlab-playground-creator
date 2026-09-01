import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")

from apps_creator.periodic.generate_news_for_public_stream import (
    prepare_generation_parser,
)
from apps_creator.periodic.src.news_generator import generate_news_parallel


class TestGenerateNewsParallel(unittest.TestCase):

    def test_prepare_generation_parser_default_num_workers(self):
        parser = prepare_generation_parser()
        args = parser.parse_args([])
        self.assertEqual(args.num_workers, 1)

    def test_prepare_generation_parser_custom_num_workers(self):
        parser = prepare_generation_parser()
        args = parser.parse_args(["--num-workers", "15"])
        self.assertEqual(args.num_workers, 15)

    def test_generate_news_parallel_empty(self):
        mock_controller = MagicMock()
        result = generate_news_parallel(
            news_controller=mock_controller,
            articles=[],
            num_workers=2,
        )
        self.assertEqual(result, [])
        mock_controller.generate_news.assert_not_called()

    @patch("apps_creator.periodic.src.news_generator.db.close_old_connections")
    def test_generate_news_parallel_batches(self, mock_close_db):
        mock_controller = MagicMock()

        # Mock articles
        mock_articles = []
        for i in range(10):
            article = MagicMock()
            article.news_url = f"https://example.com/article/{i}"
            mock_articles.append(article)

        def mock_generate(
            news_sub_page, cross_encoder_sim_model=None, ce_sim_host=None
        ):
            gen = MagicMock()
            gen.news_sub_page = news_sub_page
            return gen

        mock_controller.generate_news.side_effect = mock_generate

        results = generate_news_parallel(
            news_controller=mock_controller,
            articles=mock_articles,
            num_workers=3,
        )

        self.assertEqual(len(results), 10)
        self.assertEqual(mock_controller.generate_news.call_count, 10)
        self.assertTrue(mock_close_db.called)

    @patch("apps_creator.periodic.src.news_generator.db.close_old_connections")
    def test_generate_news_parallel_with_exceptions_and_none(self, mock_close_db):
        mock_controller = MagicMock()

        mock_articles = []
        for i in range(5):
            article = MagicMock()
            article.news_url = f"https://example.com/article/{i}"
            mock_articles.append(article)

        def mock_generate(
            news_sub_page, cross_encoder_sim_model=None, ce_sim_host=None
        ):
            if news_sub_page.news_url.endswith("/1"):
                raise RuntimeError("Failed to generate")
            elif news_sub_page.news_url.endswith("/3"):
                return None
            gen = MagicMock()
            gen.news_sub_page = news_sub_page
            return gen

        mock_controller.generate_news.side_effect = mock_generate

        results = generate_news_parallel(
            news_controller=mock_controller,
            articles=mock_articles,
            num_workers=2,
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(mock_controller.generate_news.call_count, 5)


if __name__ == "__main__":
    unittest.main()
