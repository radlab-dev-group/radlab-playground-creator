import datetime
import os
import sys
import unittest
from unittest.mock import MagicMock, call, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")

# Mock heavy modules before importing generate_articles_for_yesterday to prevent
# importing transformers / torch / cross_encoder / spacy models during unit testing.
if "creator.controllers.news" not in sys.modules:
    sys.modules["creator.controllers.news"] = MagicMock()
if "creator.controllers.clustering" not in sys.modules:
    sys.modules["creator.controllers.clustering"] = MagicMock()

from apps_creator.periodic.generate_articles_for_yesterday import (
    generate_articles_for_day,
    main,
    prepare_parser,
)


class TestGenerateArticlesForYesterday(unittest.TestCase):

    def test_prepare_parser_defaults(self):
        parser = prepare_parser()
        args = parser.parse_args([])
        self.assertEqual(args.min_cluster_count, 25)
        self.assertEqual(args.opt_cluster_count, 35)
        self.assertEqual(args.max_cluster_count, 45)
        self.assertIsNone(args.begin_date)
        self.assertIsNone(args.end_date)

    def test_prepare_parser_custom_cluster_counts(self):
        parser = prepare_parser()
        args = parser.parse_args(
            [
                "--min-cluster-count",
                "10",
                "--opt-cluster-count",
                "20",
                "--max-cluster-count",
                "30",
            ]
        )
        self.assertEqual(args.min_cluster_count, 10)
        self.assertEqual(args.opt_cluster_count, 20)
        self.assertEqual(args.max_cluster_count, 30)

    def test_prepare_parser_with_date_range(self):
        parser = prepare_parser()
        args = parser.parse_args(
            [
                "--begin-date",
                "2026-01-01",
                "--end-date",
                "2026-01-05",
            ]
        )
        self.assertEqual(args.begin_date, datetime.date(2026, 1, 1))
        self.assertEqual(args.end_date, datetime.date(2026, 1, 5))

    def test_prepare_parser_invalid_date(self):
        parser = prepare_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--begin-date", "invalid-date"])

    @patch("apps_creator.periodic.generate_articles_for_yesterday.os.path.exists")
    @patch("apps_creator.periodic.generate_articles_for_yesterday.os.unlink")
    @patch(
        "apps_creator.periodic.generate_articles_for_yesterday.InformationBrowser"
    )
    def test_generate_articles_for_day_success(
        self, mock_info_browser, mock_unlink, mock_exists
    ):
        mock_news_ctrl = MagicMock()
        mock_cl_handler = MagicMock()

        target_day = datetime.date(2026, 1, 10)
        mock_raw_news = [MagicMock()]
        mock_conv_news = [{"text": "news1", "metadata": {}}]

        mock_news_ctrl.public_get_generated_news_for_date_range.return_value = (
            mock_raw_news
        )
        mock_info_browser.convert_news_to_store_jsonl.return_value = mock_conv_news
        mock_info_browser.store_converted_news_to_jsonl_file.return_value = (
            "/tmp/news_temp.jsonl"
        )
        mock_info_browser.TEXT_COLUMN_NAME = "text"
        mock_info_browser.METADATA_COLUMN_NAME = "metadata"
        mock_exists.return_value = True

        fake_sds = MagicMock()
        fake_sds.day_to_summary = target_day
        fake_sds.when_generated = "2026-01-10"
        fake_sds.clustering = MagicMock()
        fake_clusters = []
        mock_cl_handler.to_db_objects.return_value = (fake_sds, fake_clusters)

        result = generate_articles_for_day(
            news_controller=mock_news_ctrl,
            cl_handler=mock_cl_handler,
            target_day=target_day,
            clear_dataset_if_exists=True,
        )

        mock_news_ctrl.public_get_generated_news_for_date_range.assert_called_once_with(
            begin_date=datetime.date(2026, 1, 10),
            end_date=datetime.date(2026, 1, 11),
        )
        mock_info_browser.convert_news_to_store_jsonl.assert_called_once_with(
            generated_news=mock_raw_news
        )
        mock_info_browser.store_converted_news_to_jsonl_file.assert_called_once_with(
            all_news=mock_conv_news, out_file_path=None
        )
        mock_cl_handler.clear.assert_called_once()
        mock_cl_handler.clusterer.load_dataset.assert_called_once_with(
            file_path="/tmp/news_temp.jsonl",
            text_column="text",
            metadata_column="metadata",
            input_type="jsonl",
            clear_dataset_if_exists=True,
        )
        mock_cl_handler.run.assert_called_once_with(
            generate_labels=True, generate_articles=True
        )
        mock_cl_handler.to_db_objects.assert_called_once_with(
            store_to_db=True, day_to_summary=datetime.date(2026, 1, 10)
        )
        mock_unlink.assert_called_once_with("/tmp/news_temp.jsonl")
        self.assertEqual(result, (fake_sds, fake_clusters))

    @patch(
        "apps_creator.periodic.generate_articles_for_yesterday.InformationBrowser"
    )
    def test_generate_articles_for_day_empty_news(self, mock_info_browser):
        mock_news_ctrl = MagicMock()
        mock_cl_handler = MagicMock()

        target_day = datetime.date(2026, 1, 10)
        mock_news_ctrl.public_get_generated_news_for_date_range.return_value = []
        mock_info_browser.convert_news_to_store_jsonl.return_value = []

        result = generate_articles_for_day(
            news_controller=mock_news_ctrl,
            cl_handler=mock_cl_handler,
            target_day=target_day,
        )

        self.assertIsNone(result)
        mock_cl_handler.clear.assert_not_called()
        mock_cl_handler.run.assert_not_called()

    @patch("apps_creator.periodic.generate_articles_for_yesterday.os.path.exists")
    @patch("apps_creator.periodic.generate_articles_for_yesterday.os.unlink")
    @patch(
        "apps_creator.periodic.generate_articles_for_yesterday.InformationBrowser"
    )
    def test_generate_articles_for_day_cleanup_on_exception(
        self, mock_info_browser, mock_unlink, mock_exists
    ):
        mock_news_ctrl = MagicMock()
        mock_cl_handler = MagicMock()

        target_day = datetime.date(2026, 1, 10)
        mock_news_ctrl.public_get_generated_news_for_date_range.return_value = [
            MagicMock()
        ]
        mock_info_browser.convert_news_to_store_jsonl.return_value = [
            {"text": "news1"}
        ]
        mock_info_browser.store_converted_news_to_jsonl_file.return_value = (
            "/tmp/news_temp.jsonl"
        )
        mock_info_browser.TEXT_COLUMN_NAME = "text"
        mock_info_browser.METADATA_COLUMN_NAME = "metadata"
        mock_exists.return_value = True

        mock_cl_handler.run.side_effect = RuntimeError("Clustering failed")

        with self.assertRaises(RuntimeError):
            generate_articles_for_day(
                news_controller=mock_news_ctrl,
                cl_handler=mock_cl_handler,
                target_day=target_day,
            )

        mock_unlink.assert_called_once_with("/tmp/news_temp.jsonl")

    @patch("apps_creator.periodic.generate_articles_for_yesterday.SystemController")
    @patch("apps_creator.periodic.generate_articles_for_yesterday.NewsController")
    @patch("apps_creator.periodic.generate_articles_for_yesterday.ClusteringHandler")
    @patch(
        "apps_creator.periodic.generate_articles_for_yesterday.generate_articles_for_day"
    )
    def test_main_already_running(
        self, mock_gen_day, mock_cl_handler, mock_news_ctrl, mock_system_ctrl
    ):
        mock_settings = MagicMock()
        mock_settings.doing_news_generation_for_yesterday = True
        mock_system_ctrl.get_system_settings.return_value = mock_settings

        main([])

        mock_system_ctrl.begin_public_yesterday_news_generation.assert_not_called()
        mock_gen_day.assert_not_called()

    @patch("apps_creator.periodic.generate_articles_for_yesterday.SystemController")
    @patch("apps_creator.periodic.generate_articles_for_yesterday.NewsController")
    @patch("apps_creator.periodic.generate_articles_for_yesterday.ClusteringHandler")
    @patch(
        "apps_creator.periodic.generate_articles_for_yesterday.generate_articles_for_day"
    )
    @patch("apps_creator.periodic.generate_articles_for_yesterday.datetime")
    def test_main_default_yesterday(
        self,
        mock_datetime,
        mock_gen_day,
        mock_cl_handler,
        mock_news_ctrl,
        mock_system_ctrl,
    ):
        mock_settings = MagicMock()
        mock_settings.doing_news_generation_for_yesterday = False
        mock_system_ctrl.get_system_settings.return_value = mock_settings

        fixed_today = datetime.date(2026, 3, 15)
        mock_now = MagicMock()
        mock_now.date.return_value = fixed_today
        mock_datetime.datetime.now.return_value = mock_now
        mock_datetime.timedelta = datetime.timedelta
        mock_datetime.date = datetime.date

        main([])

        mock_system_ctrl.begin_public_yesterday_news_generation.assert_called_once_with(
            mock_settings
        )
        mock_system_ctrl.end_public_yesterday_news_generation.assert_called_once_with(
            mock_settings
        )

        expected_day = datetime.date(2026, 3, 14)
        mock_gen_day.assert_called_once_with(
            news_controller=mock_news_ctrl.return_value,
            cl_handler=mock_cl_handler.return_value,
            target_day=expected_day,
        )

    @patch("apps_creator.periodic.generate_articles_for_yesterday.SystemController")
    @patch("apps_creator.periodic.generate_articles_for_yesterday.NewsController")
    @patch("apps_creator.periodic.generate_articles_for_yesterday.ClusteringHandler")
    @patch(
        "apps_creator.periodic.generate_articles_for_yesterday.generate_articles_for_day"
    )
    def test_main_date_range_iteration(
        self, mock_gen_day, mock_cl_handler, mock_news_ctrl, mock_system_ctrl
    ):
        mock_settings = MagicMock()
        mock_settings.doing_news_generation_for_yesterday = False
        mock_system_ctrl.get_system_settings.return_value = mock_settings

        main(["--begin-date", "2026-01-01", "--end-date", "2026-01-03"])

        mock_system_ctrl.begin_public_yesterday_news_generation.assert_called_once_with(
            mock_settings
        )
        mock_system_ctrl.end_public_yesterday_news_generation.assert_called_once_with(
            mock_settings
        )

        self.assertEqual(mock_gen_day.call_count, 3)
        mock_gen_day.assert_has_calls(
            [
                call(
                    news_controller=mock_news_ctrl.return_value,
                    cl_handler=mock_cl_handler.return_value,
                    target_day=datetime.date(2026, 1, 1),
                ),
                call(
                    news_controller=mock_news_ctrl.return_value,
                    cl_handler=mock_cl_handler.return_value,
                    target_day=datetime.date(2026, 1, 2),
                ),
                call(
                    news_controller=mock_news_ctrl.return_value,
                    cl_handler=mock_cl_handler.return_value,
                    target_day=datetime.date(2026, 1, 3),
                ),
            ]
        )

    @patch("apps_creator.periodic.generate_articles_for_yesterday.SystemController")
    @patch("apps_creator.periodic.generate_articles_for_yesterday.NewsController")
    @patch("apps_creator.periodic.generate_articles_for_yesterday.ClusteringHandler")
    @patch(
        "apps_creator.periodic.generate_articles_for_yesterday.generate_articles_for_day"
    )
    def test_main_single_sided_begin_date(
        self, mock_gen_day, mock_cl_handler, mock_news_ctrl, mock_system_ctrl
    ):
        mock_settings = MagicMock()
        mock_settings.doing_news_generation_for_yesterday = False
        mock_system_ctrl.get_system_settings.return_value = mock_settings

        main(["--begin-date", "2026-02-10"])

        mock_gen_day.assert_called_once_with(
            news_controller=mock_news_ctrl.return_value,
            cl_handler=mock_cl_handler.return_value,
            target_day=datetime.date(2026, 2, 10),
        )

    @patch("apps_creator.periodic.generate_articles_for_yesterday.SystemController")
    @patch("apps_creator.periodic.generate_articles_for_yesterday.NewsController")
    @patch("apps_creator.periodic.generate_articles_for_yesterday.ClusteringHandler")
    @patch(
        "apps_creator.periodic.generate_articles_for_yesterday.generate_articles_for_day"
    )
    def test_main_single_sided_end_date(
        self, mock_gen_day, mock_cl_handler, mock_news_ctrl, mock_system_ctrl
    ):
        mock_settings = MagicMock()
        mock_settings.doing_news_generation_for_yesterday = False
        mock_system_ctrl.get_system_settings.return_value = mock_settings

        main(["--end-date", "2026-02-20"])

        mock_gen_day.assert_called_once_with(
            news_controller=mock_news_ctrl.return_value,
            cl_handler=mock_cl_handler.return_value,
            target_day=datetime.date(2026, 2, 20),
        )

    @patch("apps_creator.periodic.generate_articles_for_yesterday.SystemController")
    @patch("apps_creator.periodic.generate_articles_for_yesterday.NewsController")
    @patch("apps_creator.periodic.generate_articles_for_yesterday.ClusteringHandler")
    @patch(
        "apps_creator.periodic.generate_articles_for_yesterday.generate_articles_for_day"
    )
    def test_main_handles_exception_and_releases_lock(
        self, mock_gen_day, mock_cl_handler, mock_news_ctrl, mock_system_ctrl
    ):
        mock_settings = MagicMock()
        mock_settings.doing_news_generation_for_yesterday = False
        mock_system_ctrl.get_system_settings.return_value = mock_settings

        mock_gen_day.side_effect = RuntimeError("Something unexpected")

        main(["--begin-date", "2026-01-01", "--end-date", "2026-01-02"])

        mock_system_ctrl.begin_public_yesterday_news_generation.assert_called_once_with(
            mock_settings
        )
        mock_system_ctrl.end_public_yesterday_news_generation.assert_called_once_with(
            mock_settings
        )


if __name__ == "__main__":
    unittest.main()
