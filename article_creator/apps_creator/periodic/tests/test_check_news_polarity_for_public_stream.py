import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")

from apps_creator.periodic.check_news_polarity_for_public_stream import (
    prepare_polarity_parser,
)
from creator.controllers.polarity import PolarityController


class TestCheckNewsPolarityForPublicStream(unittest.TestCase):

    def test_prepare_polarity_parser_default_num_workers(self):
        parser = prepare_polarity_parser()
        args = parser.parse_args([])
        self.assertEqual(args.num_workers, 1)

    def test_prepare_polarity_parser_custom_num_workers(self):
        parser = prepare_polarity_parser()
        args = parser.parse_args(["--num-workers", "6"])
        self.assertEqual(args.num_workers, 6)

    def test_check_polarity_3c_parallel_empty(self):
        controller = MagicMock()
        result = PolarityController.check_polarity_3c_parallel(
            controller, news_list=[], num_workers=2
        )
        self.assertEqual(result, [])
        controller.check_polarity_3c.assert_not_called()

    @patch("creator.controllers.polarity.db.close_old_connections")
    def test_check_polarity_3c_parallel_multiple_workers(self, mock_close_db):
        controller = MagicMock()
        mock_news = []
        for i in range(10):
            news = MagicMock()
            news.pk = i
            news.generated_text = f"News content {i}"
            mock_news.append(news)

        def mock_check_polarity_3c(news_list):
            news = news_list[0]
            return [{"original": news.generated_text, "polarity": "positive"}]

        controller.check_polarity_3c.side_effect = mock_check_polarity_3c

        results = PolarityController.check_polarity_3c_parallel(
            controller, news_list=mock_news, num_workers=4
        )

        self.assertEqual(len(results), 10)
        self.assertEqual(controller.check_polarity_3c.call_count, 10)
        self.assertTrue(mock_close_db.called)

    @patch("creator.controllers.polarity.db.close_old_connections")
    def test_check_polarity_3c_parallel_with_exceptions(self, mock_close_db):
        controller = MagicMock()
        mock_news = []
        for i in range(5):
            news = MagicMock()
            news.pk = i
            news.generated_text = f"News content {i}"
            mock_news.append(news)

        def mock_check_polarity_3c(news_list):
            news = news_list[0]
            if news.pk == 2:
                raise RuntimeError("LLM Router error")
            return [{"original": news.generated_text, "polarity": "neutral"}]

        controller.check_polarity_3c.side_effect = mock_check_polarity_3c

        results = PolarityController.check_polarity_3c_parallel(
            controller, news_list=mock_news, num_workers=2
        )

        # 4 successful, 1 exception handled
        self.assertEqual(len(results), 4)
        self.assertEqual(controller.check_polarity_3c.call_count, 5)
        self.assertTrue(mock_close_db.called)


if __name__ == "__main__":
    unittest.main()
