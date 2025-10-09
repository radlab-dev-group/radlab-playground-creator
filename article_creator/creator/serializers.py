from rest_framework import serializers

from creator.models import (
    NewsCategory,
    GeneratedNews,
    NewsSubPage,
    NewsMainPage,
    FullGeneratedArticles,
    Cluster,
    SampleClusterData,
    Clustering,
    SimilarClusters,
    SingleDaySummary,
)


class NewsCategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = NewsCategory
        fields = "__all__"


class NewsSubPageSerializerSimple(serializers.ModelSerializer):
    class Meta:
        model = NewsSubPage
        fields = ["news_url", "when_crawled", "num_of_generated_news"]


class NewsMainPageSerializerSimple(serializers.ModelSerializer):
    class Meta:
        model = NewsMainPage
        fields = ["pk", "main_url", "language"]


class GeneratedNewsSerializer(serializers.ModelSerializer):
    news_sub_page = NewsSubPageSerializerSimple(many=False)
    main_page_language = serializers.SerializerMethodField("get_main_page_language")

    def get_main_page_language(self, gen_news):
        return gen_news.news_sub_page.main_page.language

    class Meta:
        model = GeneratedNews
        fields = [
            "id",
            "generated_text",
            "generation_time",
            "language",
            "main_page_language",
            "model_used_to_generate_news",
            "news_sub_page",
            "polarity_3c",
            "pli_value",
            "similarity_to_original",
            "show_admin_message",
            "when_generated",
        ]


class FullGeneratedArticlesSerializer(serializers.ModelSerializer):
    class Meta:
        model = FullGeneratedArticles
        fields = [
            "article_str",
            "generation_time",
            "model_used_to_generate",
            "when_generated",
        ]


class ClusteringSerializer(serializers.ModelSerializer):
    class Meta:
        model = Clustering
        fields = "__all__"


class SampleClusterDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = SampleClusterData
        fields = "__all__"


class ClusterSerializer(serializers.ModelSerializer):
    clustering = ClusteringSerializer(many=False, read_only=True)
    sample = SampleClusterDataSerializer(many=False, read_only=True)

    class Meta:
        model = Cluster
        fields = "__all__"


class SimpleClusterSerializer(serializers.ModelSerializer):
    sample = SampleClusterDataSerializer(many=False, read_only=True)

    class Meta:
        model = Cluster
        fields = [
            "size",
            "label_str",
            "article_text",
            "news_urls",
            "news_metadata",
            "sample",
        ]


class SimilarClustersSerializer(serializers.ModelSerializer):
    source = SimpleClusterSerializer(many=False, read_only=True)
    target = SimpleClusterSerializer(many=False, read_only=True)

    class Meta:
        model = SimilarClusters
        fields = ["pk", "source", "target", "similarity_value", "similarity_metric"]


class SingleDaySummarySerializer(serializers.ModelSerializer):
    clustering = ClusteringSerializer(many=False, read_only=True)

    class Meta:
        model = SingleDaySummary
        fields = "__all__"
