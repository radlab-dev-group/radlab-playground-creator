from django.db import models
from django.contrib.auth.models import User


class SystemUser(models.Model):
    """
    Each user logged to playground
    """

    auth_user = models.ForeignKey(User, on_delete=models.PROTECT, null=False)


class PublicSSESettings(models.Model):
    news_collection_name = models.TextField(null=False)
    news_collection_description = models.TextField(null=False)
    news_collection_display_name = models.TextField(null=False)

    news_embedder_index_type = models.TextField(null=False)
    news_embedder_model_name = models.TextField(null=False)
    news_embedder_model_size = models.IntegerField(null=False)

    news_reranker_model_name = models.TextField(null=False)


class ArticlesStatistics(models.Model):
    statistics_date = models.DateTimeField(null=False, auto_now_add=True)

    pages_statistics = models.JSONField(null=False)
    polarity_statistics = models.JSONField(null=False)

    are_actual = models.BooleanField(default=False)


class SystemSettings(models.Model):
    """
    Settings for the whole system
    """

    actual_settings = models.BooleanField(null=False)
    settings_date = models.DateTimeField(null=False, auto_now_add=True)

    doing_news_indexing = models.BooleanField(null=False, default=False)
    doing_news_summarization = models.BooleanField(null=False, default=False)
    doing_news_semantic_indexing = models.BooleanField(null=False, default=False)
    doing_news_polarity_3c_check = models.BooleanField(null=False, default=False)
    doing_news_pli_check = models.BooleanField(null=False, default=False)
    doing_news_publ_stats = models.BooleanField(null=False, default=False)
    doing_news_generation_for_yesterday = models.BooleanField(
        null=False, default=False
    )
    doing_news_generation_for_yesterday_similarity = models.BooleanField(
        null=False, default=False
    )
    doing_news_information_graph = models.BooleanField(null=False, default=False)
    doing_news_information_sub_graph = models.BooleanField(null=False, default=False)

    last_news_indexing_check_begin = models.DateTimeField(null=True)
    last_news_indexing_check_end = models.DateTimeField(null=True)
    last_news_summarization_check_begin = models.DateTimeField(null=True)
    last_news_summarization_check_end = models.DateTimeField(null=True)
    last_news_semantic_indexing_check_begin = models.DateTimeField(null=True)
    last_news_semantic_indexing_check_end = models.DateTimeField(null=True)

    last_news_polarity_3c_check_begin = models.DateTimeField(null=True)
    last_news_polarity_3c_check_end = models.DateTimeField(null=True)
    last_news_pli_check_begin = models.DateTimeField(null=True)
    last_news_pli_check_end = models.DateTimeField(null=True)

    last_news_publ_stats_check_begin = models.DateTimeField(null=True)
    last_news_publ_stats_check_end = models.DateTimeField(null=True)

    last_news_generation_for_yesterday_begin = models.DateTimeField(null=True)
    last_news_generation_for_yesterday_end = models.DateTimeField(null=True)

    last_news_generation_for_yesterday_similarity_begin = models.DateTimeField(
        null=True
    )
    last_news_generation_for_yesterday_similarity_end = models.DateTimeField(
        null=True
    )

    last_news_information_graph_begin = models.DateTimeField(null=True)
    last_news_information_graph_end = models.DateTimeField(null=True)

    last_news_information_sub_graph_begin = models.DateTimeField(null=True)
    last_news_information_sub_graph_end = models.DateTimeField(null=True)

    public_stream_news_sse_settings = models.ForeignKey(
        PublicSSESettings, null=True, on_delete=models.CASCADE
    )

    last_articles_stats = models.ForeignKey(
        ArticlesStatistics, null=True, on_delete=models.CASCADE
    )
