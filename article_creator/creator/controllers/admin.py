import datetime
from typing import List, Dict, Any, AnyStr
from system.models import SystemSettings, ArticlesStatistics
from creator.models import NewsMainPage, GeneratedNews, NewsSubPage
from creator.controllers.news import NewsControllerSimple

ADMIN_MODULES = {
    "crawling": "crawling",
    "generation": "generation",
    "sse_indexing": "sse_indexing",
    "polarity_3c": "polarity_3c",
    "pli": "pli",
    "news_stats_gen": "news_stats_gen",
    "yesterday_articles": "yesterday_articles",
    "yesterday_articles_similarity": "yesterday_articles_similarity",
    "information_graph": "information_graph",
    "continuous_subgraph": "continuous_subgraph",
}

ADMIN_ACTIONS = ["restart"]


class AdminController:
    def __init__(self, add_to_db: bool = True):
        self._add_to_db = add_to_db
        self._news_controller = NewsControllerSimple()

    def get_admin_status(self) -> (List[Dict], List | Any):
        """
        Prepare administration status with:
          - indexing status
          - generation status
          - sse indexing status
        :return:
        """
        settings_response = []
        all_settings = list(SystemSettings.objects.filter(actual_settings=True))
        for setting in all_settings:
            s_set = {
                ADMIN_MODULES["crawling"]: self.__prepare_crawling_status(
                    system_settings=setting
                ),
                ADMIN_MODULES["generation"]: self.__prepare_generation_status(
                    system_settings=setting
                ),
                ADMIN_MODULES["sse_indexing"]: self.__prepare_sse_indexing_status(
                    system_settings=setting
                ),
                ADMIN_MODULES["polarity_3c"]: self.__prepare_p3c_indexing_status(
                    system_settings=setting
                ),
                ADMIN_MODULES["pli"]: self.__prepare_pli_indexing_status(
                    system_settings=setting
                ),
                ADMIN_MODULES["news_stats_gen"]: self.__prepare_news_stat_gen_status(
                    system_settings=setting
                ),
                ADMIN_MODULES[
                    "yesterday_articles"
                ]: self.__prepare_yesterday_articles_status(system_settings=setting),
                ADMIN_MODULES[
                    "yesterday_articles_similarity"
                ]: self.__prepare_yesterday_articles_similarity_status(
                    system_settings=setting
                ),
                ADMIN_MODULES[
                    "information_graph"
                ]: self.__prepare_information_graph_status(system_settings=setting),
                ADMIN_MODULES[
                    "continuous_subgraph"
                ]: self.__prepare_continuous_subgraph(system_settings=setting),
            }
            settings_response.append(s_set)

        all_settings = [s.pk for s in all_settings]
        if len(all_settings) == 1:
            all_settings = all_settings[0]
        return settings_response, all_settings

    def generate_news_statistics(
        self, make_as_last_system_stats: bool = False, settings_id=None
    ) -> (Dict[AnyStr, Dict], Dict[AnyStr, Dict], datetime.datetime):
        news_stats = {}
        polarity_statistics = {}
        cat_pages = self._news_controller.get_public_news_categories_with_pages()
        for category, main_pages in cat_pages:
            news_stats[category.name] = {}
            polarity_statistics[category.name] = {}
            for main_page in main_pages:
                page_stats, polarity_stats = self.__prepare_stats_for_page(
                    main_page=main_page
                )
                if main_page.main_url in news_stats[category.name]:
                    page_stats, polarity_stats = self.__merge_stats(
                        stats_1=news_stats[category.name][main_page.main_url],
                        stats_1_polarity=polarity_statistics[category.name][
                            main_page.main_url
                        ],
                        stats_2=page_stats,
                        stats_2_polarity=polarity_stats,
                    )
                news_stats[category.name][main_page.main_url] = page_stats

                polarity_statistics[category.name][
                    main_page.main_url
                ] = polarity_stats

        stats_datetime = None
        if make_as_last_system_stats:
            last_stats = self._add_article_stats(
                settings_id=settings_id,
                news_stats=news_stats,
                polarity_statistics=polarity_statistics,
            )
            stats_datetime = last_stats.statistics_date

        return news_stats, polarity_statistics, stats_datetime

    @staticmethod
    def get_last_news_statistics(settings_id):
        try:
            system_settings = SystemSettings.objects.get(pk=settings_id)
        except SystemSettings.DoesNotExist:
            return [], [], None
        last_stats = system_settings.last_articles_stats
        if last_stats is None:
            return [], [], None
        news_stats = last_stats.pages_statistics
        polarity_statistics = last_stats.polarity_statistics
        stats_datetime = last_stats.statistics_date
        return news_stats, polarity_statistics, stats_datetime

    @staticmethod
    def _add_article_stats(
        settings_id, news_stats: dict, polarity_statistics: dict
    ) -> ArticlesStatistics:
        ArticlesStatistics.objects.filter(are_actual=True).update(are_actual=False)
        articles_stats = ArticlesStatistics.objects.create(
            pages_statistics=news_stats,
            polarity_statistics=polarity_statistics,
            are_actual=True,
        )
        if settings_id is not None:
            SystemSettings.objects.filter(pk=settings_id).update(
                last_articles_stats=articles_stats
            )
        return articles_stats

    def do_admin_action_on_modules(
        self, action: str, module: str, settings_id
    ) -> bool:
        if not self._add_to_db:
            return False

        data_query = {}
        if module == ADMIN_MODULES["crawling"]:
            db_field_module = "news_indexing"
        elif module == ADMIN_MODULES["generation"]:
            db_field_module = "news_summarization"
        elif module == ADMIN_MODULES["sse_indexing"]:
            db_field_module = "news_semantic_indexing"
        elif module == ADMIN_MODULES["polarity_3c"]:
            db_field_module = "news_polarity_3c_check"
        elif module == ADMIN_MODULES["pli"]:
            db_field_module = "news_pli_check"
        elif module == ADMIN_MODULES["news_stats_gen"]:
            db_field_module = "news_publ_stats"
        elif module == ADMIN_MODULES["yesterday_articles"]:
            db_field_module = "news_generation_for_yesterday"
        elif module == ADMIN_MODULES["yesterday_articles_similarity"]:
            db_field_module = "news_generation_for_yesterday_similarity"
        elif module == ADMIN_MODULES["information_graph"]:
            db_field_module = "news_information_graph"
        elif module == ADMIN_MODULES["continuous_subgraph"]:
            db_field_module = "news_information_sub_graph"
        else:
            return False

        if action not in ADMIN_ACTIONS:
            return False

        if action == "restart":
            db_field_module = f"doing_{db_field_module}"
            data_query[db_field_module] = False

        if not len(data_query):
            return False

        SystemSettings.objects.filter(pk=settings_id).update(**data_query)
        return True

    @staticmethod
    def __merge_stats(
        stats_1: dict, stats_1_polarity: dict, stats_2: dict, stats_2_polarity: dict
    ):
        """
        page_stats = {
            "first_crawling_date": first_date_str,
            "last_crawling_date": last_date_str,
            "subpages_count": len(page_subpages),
            "news_count": all_page_news_count,
            "number_of_hidden_news": number_of_hidden_news,
            "number_of_visible_news": number_of_visible_news,
            "perc_of_hidden_news": (
                number_of_hidden_news / all_page_news_count
                if all_page_news_count > 0.0
                else 0.0
            ),
            "perc_of_visible_news": (
                number_of_visible_news / all_page_news_count
                if all_page_news_count > 0.0
                else 0.0
            ),
            "news_per_day": (
                all_page_news_count / num_of_diff_days
                if all_page_news_count > 0.0
                else 0.0
            ),
        }
        polarity_stats = {"3c": polarity_3c}

        :param stats_1:
        :param stats_2:
        :return:
        """
        # Merge general news stats
        merged_stats = stats_1
        for k, v in stats_2.items():
            if k == "first_crawling_date":
                merged_stats[k] = min(merged_stats[k], v)
            elif k == "last_crawling_date":
                merged_stats[k] = max(merged_stats[k], v)
            elif k == "last_crawling_date":
                merged_stats[k] = max(merged_stats[k], v)
            elif k in [
                "perc_of_hidden_news",
                "perc_of_visible_news",
                "news_per_day",
            ]:
                merged_stats[k] = 0.5 * (merged_stats[k] + v)
            else:
                merged_stats[k] += v

        # Merge polarity stats
        merged_polarity_stats = stats_1_polarity
        for k, v in stats_2.items():
            if k == "3c":
                for p_3c_label, p_3c_count in v.items():
                    merged_polarity_stats[k][p_3c_label] += p_3c_count

        return merged_stats, merged_polarity_stats

    @staticmethod
    def __prepare_crawling_status(system_settings: SystemSettings):
        return {
            "doing": system_settings.doing_news_indexing,
            "begin_date": system_settings.last_news_indexing_check_begin,
            "end_date": system_settings.last_news_indexing_check_end,
        }

    @staticmethod
    def __prepare_generation_status(system_settings: SystemSettings):
        return {
            "doing": system_settings.doing_news_summarization,
            "begin_date": system_settings.last_news_summarization_check_begin,
            "end_date": system_settings.last_news_summarization_check_end,
        }

    @staticmethod
    def __prepare_sse_indexing_status(system_settings: SystemSettings):
        return {
            "doing": system_settings.doing_news_semantic_indexing,
            "begin_date": system_settings.last_news_semantic_indexing_check_begin,
            "end_date": system_settings.last_news_semantic_indexing_check_end,
        }

    @staticmethod
    def __prepare_p3c_indexing_status(system_settings: SystemSettings):
        return {
            "doing": system_settings.doing_news_polarity_3c_check,
            "begin_date": system_settings.last_news_polarity_3c_check_begin,
            "end_date": system_settings.last_news_polarity_3c_check_end,
        }

    @staticmethod
    def __prepare_pli_indexing_status(system_settings: SystemSettings):
        return {
            "doing": system_settings.doing_news_pli_check,
            "begin_date": system_settings.last_news_pli_check_begin,
            "end_date": system_settings.last_news_pli_check_end,
        }

    @staticmethod
    def __prepare_news_stat_gen_status(system_settings: SystemSettings):
        return {
            "doing": system_settings.doing_news_publ_stats,
            "begin_date": system_settings.last_news_publ_stats_check_begin,
            "end_date": system_settings.last_news_publ_stats_check_end,
        }

    @staticmethod
    def __prepare_yesterday_articles_status(system_settings: SystemSettings):
        return {
            "doing": system_settings.doing_news_generation_for_yesterday,
            "begin_date": system_settings.last_news_generation_for_yesterday_begin,
            "end_date": system_settings.last_news_generation_for_yesterday_end,
        }

    @staticmethod
    def __prepare_yesterday_articles_similarity_status(
        system_settings: SystemSettings,
    ):
        return {
            "doing": system_settings.doing_news_generation_for_yesterday_similarity,
            "begin_date": system_settings.last_news_generation_for_yesterday_similarity_begin,
            "end_date": system_settings.last_news_generation_for_yesterday_similarity_end,
        }

    @staticmethod
    def __prepare_information_graph_status(system_settings: SystemSettings):
        return {
            "doing": system_settings.doing_news_information_graph,
            "begin_date": system_settings.last_news_information_graph_begin,
            "end_date": system_settings.last_news_information_graph_end,
        }

    @staticmethod
    def __prepare_continuous_subgraph(system_settings: SystemSettings):
        return {
            "doing": system_settings.doing_news_information_sub_graph,
            "begin_date": system_settings.last_news_information_sub_graph_begin,
            "end_date": system_settings.last_news_information_sub_graph_end,
        }

    @staticmethod
    def __prepare_stats_for_page(
        main_page: NewsMainPage,
    ) -> (Dict[AnyStr, Any], Dict[AnyStr, Any]):
        page_subpages = NewsSubPage.objects.filter(main_page=main_page).values_list(
            "when_crawled", flat=True
        )
        all_page_news = list(
            GeneratedNews.objects.filter(news_sub_page__main_page=main_page)
        )
        all_page_news_count = len(all_page_news)

        first_date_str = ""
        last_date_str = ""
        num_of_diff_days = 0
        if len(page_subpages):
            first_date = min(page_subpages)
            last_date = max(page_subpages)
            first_date_str = first_date.strftime("%Y-%m-%d %H:%M:%S")
            last_date_str = last_date.strftime("%Y-%m-%d %H:%M:%S")
            diff_date = last_date.date() - first_date.date()
            num_of_diff_days = diff_date.days

        polarity_3c = {}
        number_of_hidden_news = 0
        number_of_visible_news = 0
        for news in all_page_news:
            if news.show_news:
                number_of_visible_news += 1
            else:
                number_of_hidden_news += 1
            if news.polarity_3c is not None and len(news.polarity_3c):
                if news.polarity_3c not in polarity_3c:
                    polarity_3c[news.polarity_3c] = 0
                if news.show_news:
                    polarity_3c[news.polarity_3c] += 1

        if num_of_diff_days == 0:
            num_of_diff_days = 1

        page_stats = {
            "first_crawling_date": first_date_str,
            "last_crawling_date": last_date_str,
            "subpages_count": len(page_subpages),
            "news_count": all_page_news_count,
            "number_of_hidden_news": number_of_hidden_news,
            "number_of_visible_news": number_of_visible_news,
            "perc_of_hidden_news": (
                number_of_hidden_news / all_page_news_count
                if all_page_news_count > 0.0
                else 0.0
            ),
            "perc_of_visible_news": (
                number_of_visible_news / all_page_news_count
                if all_page_news_count > 0.0
                else 0.0
            ),
            "news_per_day": (
                number_of_visible_news / num_of_diff_days
                if number_of_visible_news > 0.0
                else 0.0
            ),
        }
        polarity_stats = {"3c": polarity_3c}

        return page_stats, polarity_stats
