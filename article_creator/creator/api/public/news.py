import json
from typing import Dict

from rest_framework.views import APIView
from rest_framework.permissions import AllowAny

from main.src.response import response_with_status
from main.src.decorators import required_params_exists, get_default_language

from creator.serializers import (
    NewsCategorySerializer,
    GeneratedNewsSerializer,
    NewsMainPageSerializerSimple,
    FullGeneratedArticlesSerializer,
)
from creator.controllers.news import (
    NewsControllerSimple,
    NewsController,
    SummaryOfDayNewsController,
)
from creator.controllers.sse_engine_public import PublicSSEController


class PublicNewsStreamCategories(APIView):
    permission_classes = (AllowAny,)

    @get_default_language
    def get(self, language, request):
        categories = NewsControllerSimple.get_public_news_categories()

        return response_with_status(
            status=True,
            language=language,
            response_body=NewsCategorySerializer(categories, many=True).data,
            error_name=None,
        )


class PublicNewsStreamCategoriesAndPages(APIView):
    permission_classes = (AllowAny,)

    @get_default_language
    def get(self, language, request):
        cat_with_pages = NewsControllerSimple.get_public_news_categories_with_pages()
        out_response = self.__prepare_response(cat_with_pages)
        return response_with_status(
            status=True,
            language=language,
            response_body=out_response,
            error_name=None,
        )

    @staticmethod
    def __prepare_response(categories_and_pages) -> Dict:
        out_response = {}
        for cat, pages in categories_and_pages:
            cat_data = dict(NewsCategorySerializer(cat, many=False).data)
            pages = NewsMainPageSerializerSimple(pages, many=True).data
            out_response[cat.name] = {
                "category_info": cat_data,
                "category_pages": pages,
            }
        return out_response


class PublicNewsAllNewsFromCategories(APIView):
    permission_classes = (AllowAny,)

    required_params = ["news_in_category"]
    optional_params = ["filter_pages", "polarity_3c", "pli_from", "pli_to"]

    n_controller = NewsControllerSimple()

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_default_language
    def get(self, language, request):
        news_in_category = int(request.data.get("news_in_category", 0))
        filter_pages = request.data.get("filter_pages", None)
        polarity_3c = request.data.get("polarity_3c", None)
        pli_from = request.data.get("pli_from", None)
        pli_to = request.data.get("pli_to", None)

        if filter_pages is not None:
            filter_pages = json.loads(request.data.get("filter_pages"))
            if not len(filter_pages):
                filter_pages = None

        gen_news_in_cat = self.n_controller.news_from_all_categories(
            news_in_category=news_in_category,
            filter_pages=filter_pages,
            polarity_3c=polarity_3c,
            pli_from=pli_from,
            pli_to=pli_to,
        )
        generated_news_ep = self.reformat_to_ep_out(gen_news_in_cat)

        return response_with_status(
            status=True,
            language=language,
            response_body=generated_news_ep,
            error_name=None,
        )

    @staticmethod
    def reformat_to_ep_out(gen_news_in_cat: Dict) -> Dict:
        n_g_n_in_c = {}
        for category, news_in_cat in gen_news_in_cat.items():
            n_g_n_in_c[category] = []
            for news in news_in_cat:
                n_g_n_in_c[category].append(
                    GeneratedNewsSerializer(news, many=False).data
                )

        return n_g_n_in_c


class PublicNewsFromCategoriesToCheckCorrectness(APIView):
    DEFAULT_MIN_SIM_TO_ORIG = 0.635
    DEFAULT_MAX_SIM_TO_ORIG = 0.9
    DEFAULT_MIN_ARTICLES_LENGTH = 130

    permission_classes = (AllowAny,)

    required_params = ["news_in_category"]
    optional_params = [
        "filter_pages",
        "min_sim_to_orig",
        "max_sim_to_orig",
        "min_article_text_length",
    ]

    n_controller = NewsControllerSimple()

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_default_language
    def get(self, language, request):
        news_in_category = int(request.data["news_in_category"])
        filter_pages = request.data.get("filter_pages", "")
        if not len(filter_pages):
            filter_pages = None
        else:
            filter_pages = json.loads(filter_pages)
            if not len(filter_pages):
                filter_pages = None

        min_sim_to_orig = request.data.get(
            "min_sim_to_orig", self.DEFAULT_MIN_SIM_TO_ORIG
        )
        max_sim_to_orig = request.data.get(
            "max_sim_to_orig", self.DEFAULT_MAX_SIM_TO_ORIG
        )
        min_article_text_length = request.data.get(
            "min_article_text_length", self.DEFAULT_MIN_ARTICLES_LENGTH
        )

        news_in_categories = self.n_controller.news_from_all_categories_to_check(
            news_in_category=news_in_category,
            filter_pages=filter_pages,
            min_sim_to_orig=min_sim_to_orig,
            max_sim_to_orig=max_sim_to_orig,
            min_article_text_length=min_article_text_length,
        )

        return response_with_status(
            status=True,
            language=language,
            response_body=PublicNewsAllNewsFromCategories.reformat_to_ep_out(
                news_in_categories
            ),
            error_name=None,
        )


class PublicArticleCreatorFromSearchResult(APIView):
    permission_classes = (AllowAny,)

    required_params = ["user_query", "news_ids"]
    optional_params = ["sse_query_response_id", "article_type"]

    news_controller = NewsController(
        add_to_db=True, models_config_path="configs/models-config.json"
    )

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_default_language
    def post(self, language, request):
        # required params
        news_ids = list(set(request.data.get("news_ids")))
        user_query = request.data.get("user_query")

        # optional params
        query_response_id = int(request.data.get("sse_query_response_id", -1))
        query_response_id = None if query_response_id < 0 else query_response_id
        new_article_type = request.data.get("article_type", None)

        # Get generated articles for identifiers
        news_for_article = NewsControllerSimple.get_news_by_ids(news_ids=news_ids)
        assert len(news_for_article) == len(news_ids)

        # Call llama-service api to generate article
        full_article_obj = self.news_controller.generate_article_full_from_news_list(
            user_query=user_query,
            news_list=news_for_article,
            new_article_type=new_article_type,
            query_response_id=query_response_id,
            model_name="google/gemma-3-12b-it",
        )

        return response_with_status(
            status=True,
            language=language,
            response_body=FullGeneratedArticlesSerializer(
                full_article_obj, many=False
            ).data,
            error_name=None,
        )


class DoActionOnGeneratedNews(APIView):
    permission_classes = (AllowAny,)

    required_params = ["news_id", "action"]

    news_controller = NewsControllerSimple()

    @required_params_exists(required_params=required_params)
    @get_default_language
    def post(self, language, request):
        news_id = request.data["news_id"]
        action = request.data["action"]

        self.news_controller.do_action_on_news(news_id=news_id, action=action)

        return response_with_status(
            status=True, language=language, response_body={}, error_name=None
        )


class PublicNewsStreamSearch(APIView):
    permission_classes = (AllowAny,)

    required_params = [
        "filter_urls",
        "text_to_search",
        "num_of_results",
        "last_days",
    ]
    optional_params = ["num_of_results"]

    news_controller = NewsControllerSimple()
    sse_controller = PublicSSEController("configs/public-sse-config.json")

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_default_language
    def post(self, language, request):
        filter_urls = json.loads(request.data.get("filter_urls"))
        text_to_search = request.data.get("text_to_search")
        # num_of_results = int(request.data.get("num_of_results"))
        last_days = int(request.data.get("last_days"))
        last_generated_news, main_urls, news_to_category, news_to_url = (
            self.news_controller.get_public_last_days_articles_from_sites(
                sites=filter_urls, last_days=last_days
            )
        )
        num_of_results = self.__number_of_results(last_days, std_news_per_day=10)

        search_news_response = self.sse_controller.search_news(
            text_to_search=text_to_search,
            num_of_results=num_of_results,
            relative_paths=main_urls,
        )
        query_response_id = search_news_response["query_response_id"]

        return response_with_status(
            status=True,
            language=language,
            response_body={
                "query_response_id": query_response_id,
                "search_result": self.__prepare_news_to_return_from_sse_api_search(
                    response_body=search_news_response,
                    last_generated_news=last_generated_news,
                    news_to_category=news_to_category,
                    news_to_url=news_to_url,
                ),
            },
            error_name=None,
        )

    @staticmethod
    def __number_of_results(last_days, std_news_per_day: int = 10):
        if last_days <= 3:
            return int(std_news_per_day * last_days)
        elif last_days <= 7:
            return int((0.7 * std_news_per_day) * last_days)
        elif last_days <= 14:
            return int((0.4 * std_news_per_day) * last_days)
        elif last_days <= 30:
            return int((0.2 * std_news_per_day) * last_days)
        else:
            return 100

    @staticmethod
    def __cutoff(general_stats: dict) -> dict:
        sorted_general_stats = {
            x: general_stats[x]
            for x in sorted(
                general_stats,
                key=lambda x: general_stats[x]["score_weighted"],
            )
            if general_stats[x]["score_weighted"] > 0.95  # 0.69
            and general_stats[x]["score"] > 0.0
        }
        return sorted_general_stats

    def __prepare_news_to_return_from_sse_api_search(
        self, response_body, last_generated_news, news_to_category, news_to_url
    ):
        general_stats = response_body["results"]["stats"]
        # print(50 * "=")
        # print("NO CUT OFF " * 5)
        # print(len(general_stats))
        general_stats = self.__select_by_score_weighted_rank_perc(general_stats)
        # print("CUT OFF " * 5)
        # print(len(general_stats))
        # print(json.dumps(general_stats, indent=2, ensure_ascii=False))
        # print(50 * "-")
        # print("CUT OFF " * 6)
        # general_stats = self.__cutoff(general_stats)
        # print(len(general_stats))
        # print(json.dumps(general_stats, indent=2, ensure_ascii=False))
        # print(50 * "=")
        see_news_relative_paths = self.__get_relative_paths(
            general_stats=general_stats
        )
        news_in_categories = {}
        for news in last_generated_news:
            news_category = news_to_category[news.pk]
            if news_category not in news_in_categories:
                news_in_categories[news_category] = []
            news_url = news_to_url[news.pk]
            if news_url in see_news_relative_paths:
                news_in_categories[news_category].append(news)
        return PublicNewsAllNewsFromCategories.reformat_to_ep_out(
            gen_news_in_cat=news_in_categories
        )

    @staticmethod
    def __select_by_score_weighted_rank_perc(
        general_stats: dict, rank_perc: float = 0.8
    ) -> dict:
        s_g_stats = sorted(
            general_stats, key=lambda x: general_stats[x]["score_weighted_scaled"]
        )

        n_stats = {}
        sm_score = 0.0
        for stat_site in s_g_stats:
            sm_score += general_stats[stat_site]["score_weighted_scaled"]
            n_stats[stat_site] = general_stats[stat_site]
            if sm_score >= rank_perc:
                break
        return n_stats

    @staticmethod
    def __get_relative_paths(general_stats: dict):
        relative_paths = []
        for _, result in general_stats.items():
            relative_paths.append(result["relative_path"])
        return list(set(relative_paths))


class GetArticlesSummaryOfDay(APIView):
    permission_classes = (AllowAny,)

    summary_controller = SummaryOfDayNewsController()

    required_params = ["date"]

    @required_params_exists(required_params=required_params)
    @get_default_language
    def get(self, language, request):
        date = request.data["date"]

        summaries = []
        if date is not None:
            summaries = self.summary_controller.get_summary_of_day(
                date=date, with_similarity=True
            )

        return response_with_status(
            status=True,
            language=language,
            response_body={"summaries": summaries},
            error_name=None,
        )
