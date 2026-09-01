import datetime
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")

from apps_creator.periodic.index_public_news_stream_to_sse import (
    prepare_indexing_parser,
)
from creator.controllers.sse_engine_public import PublicSSEController


class TestIndexPublicNewsStreamToSSE(unittest.TestCase):

    def test_prepare_indexing_parser_default_num_workers(self):
        parser = prepare_indexing_parser()
        args = parser.parse_args([])
        self.assertEqual(args.num_workers, 1)
        self.assertIsNone(args.begin_date)
        self.assertIsNone(args.end_date)

    def test_prepare_indexing_parser_custom_num_workers(self):
        parser = prepare_indexing_parser()
        args = parser.parse_args(["--num-workers", "8"])
        self.assertEqual(args.num_workers, 8)

    def test_prepare_indexing_parser_date_args(self):
        parser = prepare_indexing_parser()
        args = parser.parse_args(
            [
                "--begin-date",
                "2026-01-01",
                "--end-date",
                "2026-01-31",
            ]
        )
        self.assertEqual(args.begin_date, datetime.date(2026, 1, 1))
        self.assertEqual(args.end_date, datetime.date(2026, 1, 31))

    @patch("creator.controllers.sse_engine_public.GeneratedNews.objects.filter")
    def test_load_news_to_index_in_sse_no_dates(self, mock_filter):
        mock_qs = MagicMock()
        mock_filter.return_value = mock_qs

        result = PublicSSEController._load_news_to_index_in_sse()

        mock_filter.assert_called_once_with(
            show_news=True,
            news_sub_page__is_indexed_in_sse=False,
        )
        mock_qs.order_by.assert_called_once_with("-news_sub_page__when_crawled")

    @patch("creator.controllers.sse_engine_public.GeneratedNews.objects.filter")
    def test_load_news_to_index_in_sse_with_dates(self, mock_filter):
        mock_qs = MagicMock()
        mock_filter.return_value = mock_qs

        begin_date = datetime.date(2026, 1, 1)
        end_date = datetime.date(2026, 1, 31)

        result = PublicSSEController._load_news_to_index_in_sse(
            begin_date=begin_date,
            end_date=end_date,
        )

        mock_filter.assert_called_once_with(
            show_news=True,
            news_sub_page__is_indexed_in_sse=False,
            news_sub_page__when_crawled__gte=begin_date,
            news_sub_page__when_crawled__lt=datetime.date(2026, 2, 1),
        )
        mock_qs.order_by.assert_called_once_with("-news_sub_page__when_crawled")

    @patch.object(PublicSSEController, "_load_news_to_index_in_sse")
    def test_add_and_index_news_passes_dates(self, mock_load):
        mock_load.return_value = []
        controller = MagicMock()
        controller.JSON_MAIN_SSE_CONFIG_FIELD = (
            PublicSSEController.JSON_MAIN_SSE_CONFIG_FIELD
        )
        controller.SSE_ADD_INDEX_TEXTS_EP = (
            PublicSSEController.SSE_ADD_INDEX_TEXTS_EP
        )
        controller.SSE_LOGIN_PUBLIC_EP = PublicSSEController.SSE_LOGIN_PUBLIC_EP
        controller.data = {"collection_name": "test_collection"}
        controller._m2e2hosts = {
            PublicSSEController.JSON_MAIN_SSE_CONFIG_FIELD: {
                PublicSSEController.SSE_ADD_INDEX_TEXTS_EP: "http://example.com/index",
                PublicSSEController.SSE_LOGIN_PUBLIC_EP: "http://example.com/login",
            }
        }
        controller._load_news_to_index_in_sse = mock_load

        begin_date = datetime.date(2026, 1, 1)
        end_date = datetime.date(2026, 1, 31)

        PublicSSEController.add_and_index_news_to_sse(
            controller,
            num_workers=2,
            begin_date=begin_date,
            end_date=end_date,
        )
        mock_load.assert_called_once_with(
            begin_date=begin_date,
            end_date=end_date,
        )

    @patch.object(PublicSSEController, "_load_news_to_index_in_sse")
    def test_add_and_index_news_empty(self, mock_load):
        mock_load.return_value = []
        controller = MagicMock()
        controller.JSON_MAIN_SSE_CONFIG_FIELD = (
            PublicSSEController.JSON_MAIN_SSE_CONFIG_FIELD
        )
        controller.SSE_ADD_INDEX_TEXTS_EP = (
            PublicSSEController.SSE_ADD_INDEX_TEXTS_EP
        )
        controller.SSE_LOGIN_PUBLIC_EP = PublicSSEController.SSE_LOGIN_PUBLIC_EP
        controller.data = {"collection_name": "test_collection"}
        controller._m2e2hosts = {
            PublicSSEController.JSON_MAIN_SSE_CONFIG_FIELD: {
                PublicSSEController.SSE_ADD_INDEX_TEXTS_EP: "http://example.com/index",
                PublicSSEController.SSE_LOGIN_PUBLIC_EP: "http://example.com/login",
            }
        }
        controller._load_news_to_index_in_sse = mock_load

        PublicSSEController.add_and_index_news_to_sse(controller, num_workers=2)
        controller._call_sse_api_to_index_texts.assert_not_called()

    @patch("creator.controllers.sse_engine_public.db.close_old_connections")
    @patch("creator.controllers.sse_engine_public.NewsSubPage.objects.filter")
    def test_add_and_index_news_parallel(self, mock_subpage_filter, mock_close_db):
        controller = MagicMock()
        controller.JSON_MAIN_SSE_CONFIG_FIELD = (
            PublicSSEController.JSON_MAIN_SSE_CONFIG_FIELD
        )
        controller.SSE_ADD_INDEX_TEXTS_EP = (
            PublicSSEController.SSE_ADD_INDEX_TEXTS_EP
        )
        controller.SSE_LOGIN_PUBLIC_EP = PublicSSEController.SSE_LOGIN_PUBLIC_EP
        controller.data = {"collection_name": "test_collection"}
        controller._m2e2hosts = {
            PublicSSEController.JSON_MAIN_SSE_CONFIG_FIELD: {
                PublicSSEController.SSE_ADD_INDEX_TEXTS_EP: "http://example.com/index",
                PublicSSEController.SSE_LOGIN_PUBLIC_EP: "http://example.com/login",
            }
        }
        controller._get_ep_host.side_effect = lambda x: x
        controller._call_sse_api_to_index_texts.return_value = {
            "indexed_chunks": 1,
            "indexed_documents": 1,
        }

        mock_news = []
        for i in range(10):
            news = MagicMock()
            news.pk = i
            news.news_sub_page.pk = 100 + i
            mock_news.append(news)

        controller._load_news_to_index_in_sse.return_value = mock_news

        PublicSSEController.add_and_index_news_to_sse(controller, num_workers=4)

        self.assertEqual(controller._call_sse_api_to_index_texts.call_count, 10)
        self.assertEqual(mock_subpage_filter.call_count, 10)
        self.assertTrue(mock_close_db.called)

    @patch("creator.controllers.sse_engine_public.db.close_old_connections")
    @patch("creator.controllers.sse_engine_public.NewsSubPage.objects.filter")
    def test_add_and_index_news_with_exceptions_and_none(
        self, mock_subpage_filter, mock_close_db
    ):
        controller = MagicMock()
        controller.JSON_MAIN_SSE_CONFIG_FIELD = (
            PublicSSEController.JSON_MAIN_SSE_CONFIG_FIELD
        )
        controller.SSE_ADD_INDEX_TEXTS_EP = (
            PublicSSEController.SSE_ADD_INDEX_TEXTS_EP
        )
        controller.SSE_LOGIN_PUBLIC_EP = PublicSSEController.SSE_LOGIN_PUBLIC_EP
        controller.data = {"collection_name": "test_collection"}
        controller._m2e2hosts = {
            PublicSSEController.JSON_MAIN_SSE_CONFIG_FIELD: {
                PublicSSEController.SSE_ADD_INDEX_TEXTS_EP: "http://example.com/index",
                PublicSSEController.SSE_LOGIN_PUBLIC_EP: "http://example.com/login",
            }
        }
        controller._get_ep_host.side_effect = lambda x: x

        mock_news = []
        for i in range(5):
            news = MagicMock()
            news.pk = i
            news.news_sub_page.pk = 100 + i
            mock_news.append(news)

        def mock_call_api(
            collection_name, generated_news_to_index, ep_url, login_ep_url
        ):
            gen_news = generated_news_to_index[0]
            if gen_news.pk == 1:
                raise RuntimeError("API error")
            elif gen_news.pk == 3:
                return None
            return {"indexed_chunks": 1, "indexed_documents": 1}

        controller._call_sse_api_to_index_texts.side_effect = mock_call_api
        controller._load_news_to_index_in_sse.return_value = mock_news

        PublicSSEController.add_and_index_news_to_sse(controller, num_workers=2)

        self.assertEqual(controller._call_sse_api_to_index_texts.call_count, 5)
        # 3 successful updates (pk 0, 2, 4)
        self.assertEqual(mock_subpage_filter.call_count, 3)


if __name__ == "__main__":
    unittest.main()
