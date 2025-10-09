from django.urls import path

from main.src.constants import prepare_api_url

from system.api.public.admin import (
    GetAdminNewsStatistics,
    GetAdminOptions,
    DoAdminActionOnModule,
)

urlpatterns = [
    path(
        prepare_api_url("public/admin_system_status"),
        GetAdminOptions.as_view(),
        name="public_list_categories",
    ),
    path(
        prepare_api_url("public/admin_news_statistics"),
        GetAdminNewsStatistics.as_view(),
        name="public_list_categories",
    ),
    path(
        prepare_api_url("public/do_admin_action_on_module"),
        DoAdminActionOnModule.as_view(),
        name="public_do_admin_action_on_module",
    ),
]
