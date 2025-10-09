from django.urls import path

from main.src.constants import prepare_api_url

from creator.api.public.news import (
    PublicNewsStreamCategories,
    PublicNewsStreamCategoriesAndPages,
    PublicNewsAllNewsFromCategories,
    PublicNewsFromCategoriesToCheckCorrectness,
    PublicArticleCreatorFromSearchResult,
    DoActionOnGeneratedNews,
    PublicNewsStreamSearch,
    GetArticlesSummaryOfDay,
)


urlpatterns = [
    path(
        prepare_api_url("public/news_categories"),
        PublicNewsStreamCategories.as_view(),
        name="public_list_categories",
    ),
    path(
        prepare_api_url("public/news_categories_with_pages"),
        PublicNewsStreamCategoriesAndPages.as_view(),
        name="public_list_categories",
    ),
    path(
        prepare_api_url("public/last_news"),
        PublicNewsAllNewsFromCategories.as_view(),
        name="public_list_last_news",
    ),
    path(
        prepare_api_url("public/last_news_to_check_correctness"),
        PublicNewsFromCategoriesToCheckCorrectness.as_view(),
        name="public_list_last_news_to_check_correctness",
    ),
    path(
        prepare_api_url("public/generate_article_from_news_list"),
        PublicArticleCreatorFromSearchResult.as_view(),
        name="public_generate_article_from_news_list",
    ),
    path(
        prepare_api_url("public/do_action_on_news"),
        DoActionOnGeneratedNews.as_view(),
        name="public_do_action_on_news",
    ),
    path(
        prepare_api_url("public/search_news"),
        PublicNewsStreamSearch.as_view(),
        name="public_do_action_on_news",
    ),
    path(
        prepare_api_url("public/articles_summary_of_day"),
        GetArticlesSummaryOfDay.as_view(),
        name="articles_summary_of_day",
    ),
]
