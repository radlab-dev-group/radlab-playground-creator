from django.db import models

from system.models import SystemUser


# ==================================================================================
# Main tables, used when crawling specified sites.
#  - configuration for each site
#  - hashed content
#  - subpages content
class NewsCategory(models.Model):
    """
    pk: auto-id
    """

    name = models.TextField(unique=True)
    display_name = models.TextField(null=False, blank=False)
    description = models.TextField(null=False, blank=False)

    order = models.IntegerField(null=False, default=0)


class NewsMainPage(models.Model):
    """
    pk: auto-id
    """

    main_url = models.URLField(null=False)
    begin_crawling_url = models.URLField(null=False)
    category = models.ForeignKey(NewsCategory, null=False, on_delete=models.PROTECT)
    min_news_depth = models.IntegerField(null=False)

    last_check = models.DateTimeField(null=True)
    index_page = models.BooleanField(null=False, default=False)
    index_page_sse = models.BooleanField(null=False, default=False)
    prepare_news = models.BooleanField(null=False, default=False)
    show_news = models.BooleanField(null=False, default=False)

    use_heuristic_to_extract_news_content = models.BooleanField(
        null=False, default=False
    )
    use_heuristic_to_search_news_content = models.BooleanField(
        null=False, default=False
    )
    use_heuristic_to_clear_news_content = models.BooleanField(
        null=False, default=False
    )
    news_link_starts_with_main_url = models.BooleanField(null=False, default=False)

    include_paths_with = models.JSONField(null=False, default=list)
    exclude_paths_with = models.JSONField(null=False, default=list)
    news_url_ends_with = models.JSONField(null=False, default=list)

    single_news_content_tag_name = models.TextField(null=True)
    single_news_content_tag_attr_name = models.TextField(null=True)
    single_news_content_tag_attr_value = models.TextField(null=True)

    remove_n_last_elems = models.IntegerField(null=True, default=0)

    user = models.ForeignKey(SystemUser, null=True, on_delete=models.PROTECT)

    language = models.TextField(null=True)


class MainNewsPageContent(models.Model):
    """
    pk: auto-id
    """

    main_page = models.ForeignKey(NewsMainPage, null=False, on_delete=models.PROTECT)

    when_crawled = models.DateTimeField(null=False, auto_now=True)

    #
    content_hash = models.TextField(null=False, blank=False)
    html_content = models.TextField(null=False, blank=False)
    extracted_urls = models.JSONField(null=False, default=list)


class NewsSubPage(models.Model):
    """
    pk: auto-id
    """

    news_url = models.TextField(null=False)
    when_crawled = models.DateTimeField(null=False, auto_now=True)
    page_content_html = models.TextField(null=False)
    page_content_txt = models.TextField(null=False)

    skip_subpage = models.BooleanField(null=False, default=False)
    has_generated_news = models.BooleanField(null=False, default=False)
    is_indexed_in_sse = models.BooleanField(null=False, default=False)

    num_of_generated_news = models.IntegerField(null=False, default=0)

    main_page = models.ForeignKey(NewsMainPage, null=False, on_delete=models.PROTECT)

    class Meta:
        unique_together = ("news_url", "main_page")


# ==================================================================================
# Generated news:
#  - public stram table
#  - public creator table
class GeneratedNews(models.Model):
    """
    pk: auto-id
    """

    generated_text = models.TextField(null=False)
    generation_time = models.DurationField(null=False)
    language = models.TextField(null=True)

    model_used_to_generate_news = models.TextField(null=False)

    show_news = models.BooleanField(null=False, default=True)
    news_sub_page = models.ForeignKey(
        NewsSubPage, null=False, on_delete=models.PROTECT
    )

    polarity_3c = models.CharField(null=True, max_length=32)
    polarity_3c_logits = models.JSONField(null=True)

    pli_value = models.FloatField(null=True)

    metadata = models.JSONField(null=False, default=dict)

    when_generated = models.DateTimeField(null=False, auto_now=True)

    similarity_to_original = models.FloatField(null=True)

    show_admin_message = models.BooleanField(null=False, default=True)


class FullGeneratedArticles(models.Model):
    """
    pk: auto-id
    """

    article_str = models.TextField(null=False)
    generation_time = models.DurationField(null=False)
    model_used_to_generate = models.TextField(null=False)
    when_generated = models.DateTimeField(null=False, auto_now=True)

    user_query = models.TextField(null=False)
    based_on_news = models.JSONField(null=False)

    article_type = models.TextField(null=True)
    sse_query_response_id = models.TextField(null=True)

    language = models.TextField(null=True)

    is_active = models.BooleanField(null=False, default=True)


# ==================================================================================
# Everything connected with clustering:
#  - clusters
#  - similarity
#  - the single-day summary (based on the cluster)
class Clustering(models.Model):
    """
    pk: auto-id
    """

    # Genai model used to generate labels
    genai_labels_model = models.TextField(null=False)
    genai_labels_prompt = models.TextField(null=True)

    # Genai model used to generate article
    genai_article_model = models.TextField(null=False)
    genai_article_prompt = models.TextField(null=True)

    clustering_options = models.JSONField(null=False)
    clustering_method = models.TextField(null=False)

    reducer_method = models.TextField(null=False)
    reducer_optimizer = models.TextField(null=False)
    reducer_similarity = models.TextField(null=False)

    is_active = models.BooleanField(null=False, default=True)


class SampleClusterData(models.Model):
    """
    pk: auto-id
    """

    # Sample used as information
    size = models.IntegerField(null=False)
    news_urls = models.JSONField(null=False)
    news_metadata = models.JSONField(null=False)

    is_active = models.BooleanField(null=False, default=True)


class Cluster(models.Model):
    """
    pk: auto-id
    """

    # Integer label -> from clustering model
    label = models.IntegerField(null=False)

    # Size of the cluster, number of texts into the cluster
    size = models.IntegerField(null=False)

    # Label as str, which genai may generate
    label_str = models.TextField(null=False)

    # Additional info to show
    description = models.TextField(null=True)

    # Boolean value that tells if this cluster is an outlier
    is_outlier = models.BooleanField(null=False)

    # Text of a generated article based on the connected urls
    article_text = models.TextField(null=False)

    # Statistics of the full cluster
    stats = models.JSONField(null=False)

    # List of the all urls connected with the cluster
    news_urls = models.JSONField(null=False)
    # The list of metadata connected with each news url
    # length of metadata has to be the same as news_urls
    news_metadata = models.JSONField(null=False)

    sample_indices = models.JSONField(null=True)

    # Connection with sample data
    sample = models.ForeignKey(
        SampleClusterData, null=True, on_delete=models.PROTECT
    )

    # Connection with clustering
    clustering = models.ForeignKey(Clustering, null=False, on_delete=models.PROTECT)

    is_active = models.BooleanField(null=False, default=True)

    # Similarity to "next clusters" at the timeline
    has_next_similarity = models.BooleanField(null=False, default=False)
    # Similarity to "previous clusters" at the timeline
    has_prev_similarity = models.BooleanField(null=False, default=False)


class SimilarClusters(models.Model):
    source = models.ForeignKey(
        Cluster, null=False, on_delete=models.PROTECT, related_name="source_cluster"
    )
    target = models.ForeignKey(
        Cluster, null=False, on_delete=models.PROTECT, related_name="target_cluster"
    )

    similarity_value = models.FloatField(null=False)
    similarity_metric = models.TextField(null=False)

    similarity_model = models.TextField(null=True)

    is_active = models.BooleanField(null=False, default=True)

    # Connection with clustering
    clustering = models.ForeignKey(Clustering, null=False, on_delete=models.PROTECT)

    class Meta:
        unique_together = ("source", "target", "clustering")


class SingleDaySummary(models.Model):
    """
    pk: auto-id
    """

    day_to_summary = models.DateField(null=False)

    when_generated = models.DateTimeField(null=False, auto_now=True)
    clustering = models.ForeignKey(Clustering, null=False, on_delete=models.PROTECT)

    is_active = models.BooleanField(null=False, default=True)


# ==================================================================================
# Information graph definition. Handling clusters as graphs of similar clusters.
# Specification of:
#  - base graph (abstraction)
#  - continuous information graphs
# ----------------------------------------------------------------------------------
# Base graph
class BaseInformationGraph(models.Model):
    # pk: auto-id

    when_created = models.DateTimeField(null=False, auto_now=True)
    description = models.TextField(null=True)

    is_active = models.BooleanField(null=False, default=True)
    has_sub_graphs = models.BooleanField(null=False, default=True)

    graph_directory = models.TextField(null=True)
    graph_file_name = models.TextField(null=True)

    info = models.JSONField(null=True, default=dict)
    summary = models.JSONField(null=True, default=dict)

    class Meta:
        abstract = True


class BaseInformationSubGraph(models.Model):
    # pk: auto-id

    graph = models.ForeignKey(
        BaseInformationGraph, null=False, on_delete=models.PROTECT
    )

    when_created = models.DateTimeField(null=False, auto_now=True)

    sub_graph_directory = models.TextField(null=True)
    sub_graph_file_name = models.TextField(null=True)

    name = models.TextField(null=True)
    comment = models.TextField(null=True)
    article = models.TextField(null=True)

    type_str = models.TextField(null=False, default="")

    info = models.JSONField(null=True)

    is_active = models.BooleanField(null=False, default=True)

    class Meta:
        abstract = True


class BaseClustersInInformationSubGraph(models.Model):
    # pk: auto-id

    sub_graph = models.ForeignKey(
        BaseInformationSubGraph, null=False, on_delete=models.PROTECT
    )

    when_created = models.DateTimeField(null=False, auto_now=True)
    cluster = models.ForeignKey(Cluster, null=False, on_delete=models.PROTECT)

    class Meta:
        abstract = True


# ----------------------------------------------------------------------------------
# continuous graph
class ContinuousInformationGraph(BaseInformationGraph):
    """
    Continuous Information Graph -- specification for BaseGraph
    """

    ...


class ContinuousInformationSubGraph(BaseInformationSubGraph):
    """
    Single subgraph, extracted based on the specific definition.
    """

    # pk: auto-id

    graph = models.ForeignKey(
        ContinuousInformationGraph,
        null=False,
        on_delete=models.PROTECT,
        related_name="continuous_sub_graph",
    )

    label = models.TextField(null=False)
    label_str = models.TextField(null=False)


class ClustersInContinuousInformationSubGraph(BaseClustersInInformationSubGraph):
    """
    Single Cluster object belonging to the continuous information subgraph.
    """

    # pk: auto-id

    sub_graph = models.ForeignKey(
        ContinuousInformationSubGraph,
        null=False,
        on_delete=models.PROTECT,
        related_name="clusters_in_continuous_sub_graph",
    )

    class Meta:
        unique_together = ("cluster", "sub_graph")
