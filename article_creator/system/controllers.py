from typing import Dict
from django.utils import timezone

from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password

from system.models import (
    SystemUser,
    SystemSettings,
    PublicSSESettings,
)


class SystemController:
    def __init__(self):
        pass

    @staticmethod
    def proper_user_name(email: str) -> str:
        return email.replace("@", "_").replace(".", "").replace(" ", "")

    @staticmethod
    def add_system_user(
        email: str,
        password: str,
    ) -> SystemUser:
        user_name = SystemController.proper_user_name(email=email)
        user, created = User.objects.get_or_create(username=user_name)

        if created:
            user.email = email
            user.password = make_password(password)
            user.is_active = True
            user.is_staff = False
            user.is_superuser = False
            user.save()

        plg_user, created = SystemUser.objects.get_or_create(auth_user=user)
        return plg_user

    @staticmethod
    def get_system_user(email: str) -> SystemUser | None:
        """
        User is represented as properly converted email. Func. `proper_user_name`
        :param email: user email
        :return: Object of logged user or None
        """
        user_name = SystemController.proper_user_name(email=email)
        try:
            return SystemUser.objects.get(auth_user__username=user_name)
        except SystemUser.DoesNotExist:
            return None

    @staticmethod
    def get_system_settings() -> SystemSettings:
        all_settings = SystemSettings.objects.filter(actual_settings=True)
        if not len(all_settings):
            return SystemSettings.objects.create(actual_settings=True)
        if len(all_settings) > 2:
            raise Exception(f"There are {len(all_settings)} settings!")
        return all_settings[0]

    @staticmethod
    def get_public_sse_news_stream_settings(
        system_settings: SystemSettings, sse_settings_dict: Dict[str, str]
    ) -> PublicSSESettings:
        p_settings = system_settings.public_stream_news_sse_settings
        if p_settings is not None:
            return p_settings

        p_settings = PublicSSESettings.objects.create(
            news_collection_name=sse_settings_dict["news_collection_name"],
            news_collection_description=sse_settings_dict[
                "news_collection_description"
            ],
            news_collection_display_name=sse_settings_dict[
                "news_collection_display_name"
            ],
            news_embedder_index_type=sse_settings_dict["news_embedder_index_type"],
            news_embedder_model_name=sse_settings_dict["news_embedder_model_name"],
            news_embedder_model_size=sse_settings_dict["news_embedder_model_size"],
            news_reranker_model_name=sse_settings_dict["news_reranker_model_name"],
        )
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            public_stream_news_sse_settings=p_settings
        )
        return p_settings

    @staticmethod
    def begin_public_news_indexing(system_settings: SystemSettings) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_indexing=True, last_news_indexing_check_begin=timezone.now()
        )

    @staticmethod
    def end_public_news_indexing(system_settings: SystemSettings) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_indexing=False, last_news_indexing_check_end=timezone.now()
        )

    @staticmethod
    def begin_public_news_indexing_sse(system_settings: SystemSettings) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_semantic_indexing=True,
            last_news_semantic_indexing_check_begin=timezone.now(),
        )

    @staticmethod
    def end_public_news_indexing_sse(system_settings: SystemSettings) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_semantic_indexing=False,
            last_news_semantic_indexing_check_end=timezone.now(),
        )

    @staticmethod
    def begin_public_news_generation(system_settings: SystemSettings) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_summarization=True,
            last_news_summarization_check_begin=timezone.now(),
        )

    @staticmethod
    def end_public_news_generation(system_settings: SystemSettings) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_summarization=False,
            last_news_summarization_check_end=timezone.now(),
        )

    @staticmethod
    def begin_public_yesterday_news_generation(
        system_settings: SystemSettings,
    ) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_generation_for_yesterday=True,
            last_news_generation_for_yesterday_begin=timezone.now(),
        )

    @staticmethod
    def end_public_yesterday_news_generation(
        system_settings: SystemSettings,
    ) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_generation_for_yesterday=False,
            last_news_generation_for_yesterday_end=timezone.now(),
        )

    @staticmethod
    def begin_public_yesterday_news_generation_similarity(
        system_settings: SystemSettings,
    ) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_generation_for_yesterday_similarity=True,
            last_news_generation_for_yesterday_similarity_begin=timezone.now(),
        )

    @staticmethod
    def end_public_yesterday_news_generation_similarity(
        system_settings: SystemSettings,
    ) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_generation_for_yesterday_similarity=False,
            last_news_generation_for_yesterday_similarity_end=timezone.now(),
        )

    @staticmethod
    def begin_doing_information_graph(
        system_settings: SystemSettings,
    ) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_information_graph=True,
            last_news_information_graph_begin=timezone.now(),
        )

    @staticmethod
    def end_doing_information_graph(
        system_settings: SystemSettings,
    ) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_information_graph=False,
            last_news_information_graph_end=timezone.now(),
        )

    @staticmethod
    def begin_doing_information_sub_graph(
        system_settings: SystemSettings,
    ) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_information_sub_graph=True,
            last_news_information_sub_graph_begin=timezone.now(),
        )

    @staticmethod
    def end_doing_information_sub_graph(
        system_settings: SystemSettings,
    ) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_information_sub_graph=False,
            last_news_information_sub_graph_end=timezone.now(),
        )

    @staticmethod
    def begin_public_news_polarity_check(system_settings: SystemSettings) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_polarity_3c_check=True,
            last_news_polarity_3c_check_begin=timezone.now(),
        )

    @staticmethod
    def end_public_news_polarity_check(system_settings: SystemSettings) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_polarity_3c_check=False,
            last_news_polarity_3c_check_end=timezone.now(),
        )

    @staticmethod
    def begin_public_news_pli_check(system_settings: SystemSettings) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_pli_check=True,
            last_news_pli_check_begin=timezone.now(),
        )

    @staticmethod
    def end_public_news_pli_check(system_settings: SystemSettings) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_pli_check=False,
            last_news_pli_check_end=timezone.now(),
        )

    @staticmethod
    def begin_public_make_news_stats(system_settings: SystemSettings) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_publ_stats=True,
            last_news_publ_stats_check_begin=timezone.now(),
        )

    @staticmethod
    def end_public_make_news_stats(system_settings: SystemSettings) -> None:
        SystemSettings.objects.filter(pk=system_settings.pk).update(
            doing_news_publ_stats=False,
            last_news_publ_stats_check_end=timezone.now(),
        )
