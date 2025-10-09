from rest_framework.views import APIView
from rest_framework.permissions import AllowAny

from main.src.decorators import get_default_language, required_params_exists
from main.src.response import response_with_status

from creator.controllers.admin import AdminController


class GetAdminOptions(APIView):
    permission_classes = (AllowAny,)

    admin_controller = AdminController()

    @get_default_language
    def get(self, language, request):
        admin_status, settings = self.admin_controller.get_admin_status()

        return response_with_status(
            status=True,
            language=language,
            response_body={"status": admin_status, "settings": settings},
            error_name=None,
        )


class GetAdminNewsStatistics(APIView):
    permission_classes = (AllowAny,)

    admin_controller = AdminController()
    required_params = ["settings_id"]
    optional_params = ["get_last_stats"]

    @required_params_exists(
        required_params=required_params, optional_params=optional_params
    )
    @get_default_language
    def get(self, language, request):
        settings_id = request.data["settings_id"]
        get_last_stats = request.data.get("get_last_stats", None)
        if get_last_stats is not None:
            page_stats, polarity_stats, stats_datetime = (
                self.admin_controller.get_last_news_statistics(settings_id)
            )
        else:
            page_stats, polarity_stats, stats_datetime = (
                self.admin_controller.generate_news_statistics(
                    make_as_last_system_stats=True, settings_id=settings_id
                )
            )

        if stats_datetime is not None:
            stats_datetime = stats_datetime.strftime("%Y-%m-%d %H:%M:%S")

        return response_with_status(
            status=True,
            language=language,
            response_body={
                "news_stats": page_stats,
                "polarity_stats": polarity_stats,
                "stats_datetime": stats_datetime,
            },
            error_name=None,
        )


class DoAdminActionOnModule(APIView):
    permission_classes = (AllowAny,)
    required_params = ["action", "module", "settings_id"]

    admin_controller = AdminController()

    @required_params_exists(required_params=required_params)
    @get_default_language
    def post(self, language, request):
        action = request.data["action"]
        module = request.data["module"]
        settings_id = request.data["settings_id"]
        action_status = self.admin_controller.do_admin_action_on_modules(
            action=action, module=module, settings_id=settings_id
        )

        return response_with_status(
            status=True,
            language=language,
            response_body={
                "settings_id": settings_id,
                "action": action,
                "module": module,
                "status": action_status,
            },
            error_name=None,
        )
