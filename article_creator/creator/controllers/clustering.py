import torch
import numpy
import random
import datetime
import requests

from tqdm import tqdm
from django.db.models import Q
from django.utils import timezone

from main.src.constants import get_logger

from clusterer.clustering.config import ClustererConfig
from clusterer.clustering.clusterer import RdlClusterer
from clusterer.dataset.embedding_dataset import EmbeddingsDatasetHandler

from creator.models import (
    Cluster as ClusterDB,
    SampleClusterData,
    Clustering,
    SimilarClusters,
    SingleDaySummary,
)


class ClusteringPrompts:
    GENERATE_LABEL = """
You are an agent who helps to come up with a category name (in Polish) from the submitted texts.
The category name should capture the essence of the submitted texts as best as possible. 
The name can be a short sentence. Apart from the category name, do not write anything.
Do not add anything from yourself. Be creative when coming up with a name.
"""

    GENERATE_LABEL_PL = """
Jesteś agentem, który pomaga w wymysleniu nazwy kategorii z przesłanych tekstów.
Nazwa kategorii powinna jak najlepiej oddawać istotę przesłanych tekstów. 
Nazwa może być krótkim zdaniem. Oprócz nazwy kategorii nie pisz nic.
Nic od siebie nie dodawaj. Bądź kreatywny przy wymyślaniu nazwy. 
"""

    PREPARE_ARTICLE = """
You are an agent whose job is to write short (about an A4 page)
articles, summarizing texts/news sent to you by a user.
The news that the user sends you is from one day. You write an overview of
articles that came out today. Do not write what day it is.
Also, don't state that the article is written based on the 
based on the content provided. Write the article in Polish.
"""

    PREPARE_ARTICLE_PL = """
Jesteś agentem, którego zadaniem jest pisanie krótkich (około strony A4)
artykułów, podsumowujących teksty/newsy przesłane przez użytkownika.
Newsy, które przesyła Ci użytkownik są z jednego dnia. Piszesz przegląd
artykułów, które ukazały się dziś. Nie pisz jakiego to dnia.
Nie podawaj również informacji, że artykuł pisany jest na 
podstawie dostarczonych treści. Artykuł napisz po polsku.
"""

    CHECK_SPELLING = """
Popraw literówki w przesłanym przez użytkownika tekście.
W odpowiedzi podaj tylko artykuł z poprawionymi błędami językowymi.
Nie dodawaj nic od siebie, nie zmieniaj sensu przesłanego tekstu.
"""


class Cluster:
    TITLE_SIZE_MD = "### "

    def __init__(self, label: int):
        self.label = label
        self.generated_label = ""
        self.generated_article = ""

        self.texts = []
        self.proper_texts = []
        self.metadata_list = []

        self.random_texts = []
        self.random_proper_texts = []
        self.random_metadata_list = []

        self.news_pk = []
        self.news_urls = []
        self.random_news_pk = []
        self.random_news_urls = []

        self.random_indices = []

        self.stats = {}

    def append(self, text: str, proper_text: str, metadata: dict):
        self.texts.append(text)
        self.proper_texts.append(proper_text)
        self.metadata_list.append(metadata)

        self.news_urls.append(metadata.get("news_url"))
        self.news_pk.append(metadata.get("news_pk"))

    def rand_texts(self, num_of_texts: int):
        assert len(self.texts)

        if len(self.texts) < num_of_texts:
            num_of_texts = len(self.texts)

        indices = [i for i in range(num_of_texts)]
        random.shuffle(indices)

        self.random_indices = indices[:num_of_texts]
        self.__prepare_rand_texts()

    def prepare_stats(self):
        avg_md = ["pli_value"]
        enum_md = ["language", "polarity_3c", "source"]

        stats = {}
        for md in self.metadata_list:
            for md_name, md_value in md.items():
                if md_value is None:
                    continue

                if md_name in avg_md:
                    if md_name not in stats:
                        stats[md_name] = []
                    stats[md_name].append(md_value)
                elif md_name in enum_md:
                    if md_name not in stats:
                        stats[md_name] = {}
                    if md_value not in stats[md_name]:
                        stats[md_name][md_value] = 0
                    stats[md_name][md_value] += 1

        for avg_md_name in avg_md:
            stats[avg_md_name] = sum(stats[avg_md_name]) / len(stats[avg_md_name])

        stats["num_of_texts"] = len(self.texts)

        self.stats = stats
        return self.stats

    def generate_label_with_genai(self, host: str, ep: str, payload: dict):
        self.generated_label = ""

        response = self.__request_post_extended_conversation_with_data(
            host=host,
            ep=ep,
            payload=payload,
            data=self.random_proper_texts,
            prompt=ClusteringPrompts.GENERATE_LABEL,
        )

        if not response.ok:
            return

        information_name = response.json()["response"]
        generated_label = self.__clear_generated_information_label(
            label=information_name
        )
        self.generated_label = generated_label.strip()

    def generate_article(
        self, host: str, ep: str, payload: dict, check_spelling: bool = True
    ):
        self.generated_article = None

        response = self.__request_post_extended_conversation_with_data(
            host=host,
            ep=ep,
            payload=payload,
            data=self.random_proper_texts,
            prompt=ClusteringPrompts.PREPARE_ARTICLE,
        )

        if not response.ok:
            return

        article_text = response.json()["response"]
        if check_spelling:
            article_text = self.__check_and_repair_spelling(
                text=article_text, host=host, ep=ep, payload=payload
            )

        article_text = self.__clear_generated_article(article_text=article_text)
        self.generated_article = article_text.strip()

    def to_cluster_db(self, to_db: bool, clustering: Clustering) -> ClusterDB:
        sample_cluster_data = SampleClusterData(
            size=len(self.random_texts),
            news_urls=self.random_news_urls,
            news_metadata=self.random_metadata_list,
        )
        if to_db:
            sample_cluster_data.save()

        cluster_db = ClusterDB(
            label=self.label,
            size=len(self.texts),
            label_str=self.generated_label,
            is_outlier=self.label == -1,
            article_text=self.generated_article,
            stats=self.stats,
            news_urls=self.news_urls,
            news_metadata=self.metadata_list,
            sample_indices=self.random_indices,
            sample=sample_cluster_data,
            clustering=clustering,
        )
        if to_db:
            cluster_db.save()

        return cluster_db

    def __prepare_rand_texts(self):
        self.random_texts = []
        self.random_proper_texts = []
        self.random_metadata_list = []
        for i in self.random_indices:
            self.random_texts.append(self.texts[i])
            self.random_proper_texts.append(self.proper_texts[i])
            self.random_metadata_list.append(self.metadata_list[i])
            self.random_news_pk.append(self.news_pk[i])
            self.random_news_urls.append(self.news_urls[i])

    def __check_and_repair_spelling(
        self, text: str, host: str, ep: str, payload: dict
    ) -> str:

        response = self.__request_post_extended_conversation_with_data(
            host=host,
            ep=ep,
            payload=payload,
            data=[text],
            prompt=ClusteringPrompts.CHECK_SPELLING,
        )

        if not response.ok:
            return text

        repaired_text = response.json()["response"]
        return repaired_text.strip()

    @staticmethod
    def __request_post_extended_conversation_with_data(
        host: str, ep: str, payload: dict, data: list[str], prompt: str
    ):
        ep = ep.strip("/").strip()
        host = host.strip("/").strip()

        ep_body = payload.copy()
        ep_body["system_prompt"] = prompt.strip()
        ep_body["user_last_statement"] = "\n".join(data).strip()

        ep_url = f"{host}/{ep}"
        response = requests.post(
            ep_url, json=ep_body, headers={"Content-Type": "application/json"}
        )

        return response

    def __clear_generated_article(self, article_text: str) -> str:
        article_text = self.__replace_predefined_phrases(article_text=article_text)
        article_text = self.__reconstruct_title(article_text=article_text)
        return article_text.strip()

    @staticmethod
    def __clear_generated_information_label(label: str) -> str:
        cl_label = label.replace("*", "")
        return cl_label.strip()

    def __replace_predefined_phrases(self, article_text: str) -> str:
        clr_phrases = [
            ["Dziśnie", "Dzisie"],
            ["## ", self.TITLE_SIZE_MD],
            [" $", " \$"],
        ]
        for ch_from, ch_to in clr_phrases:
            article_text = article_text.replace(ch_from, ch_to)
        return article_text.strip()

    def __reconstruct_title(self, article_text: str) -> str:
        title_not_ends_with = [".", "!", "?", ",", ":", ";"]
        if article_text.startswith(self.TITLE_SIZE_MD) or not len(article_text):
            return article_text

        # First line as a title
        article_text = article_text.strip()
        title = article_text.split("\n")[0].strip()
        if title[-1] in title_not_ends_with:
            # When the first line is the proper sentence.
            title = self.__rand_predefined_title(titles=None)
        article_text = (
            self.TITLE_SIZE_MD + title.strip() + "\n\n" + article_text.strip()
        )
        return article_text.strip()

    @staticmethod
    def __rand_predefined_title(titles: list or None = None):
        predefined_titles = [
            "Przegląd informacji z dziś",
            "Przegląd informacji z dnia",
            "Podsumowanie informacji z dziś",
        ]
        if titles is not None and len(titles):
            predefined_titles = titles
        return random.choice(predefined_titles)


class ClusteringHandler:
    MAX_TEXTS_TO_API = 15
    MAX_TEXT_CHARS_LENGTH = 700

    def __init__(
        self,
        clustering_config_path: str,
        min_cluster_count: int,
        opt_cluster_count: int,
        max_cluster_count: int,
    ):
        self.prepared_clusters = False

        self.config = ClustererConfig(config_file_path=clustering_config_path)
        self.clusterer = RdlClusterer(
            embedder_path=self.config.embedder_path,
            embedder_input_size=self.config.embedder_input_size,
            method=self.config.method,
            reduction=self.config.reduction,
            reducer_optim=self.config.reducer_optim,
            reducer_sim_metric=self.config.reducer_sim_metric,
            device=self.config.device,
            load_model=True,
            use_reduced_dataset=self.config.use_reduced_dataset,
            clustering_params=self.config.clustering_params,
            min_cluster_count=min_cluster_count,
            opt_cluster_count=opt_cluster_count,
            max_cluster_count=max_cluster_count,
        )

        self.clusters_objects = {}

        # LLMS service config
        self.llms_service_host = ""
        self.prepare_labels_ep = ""
        self.prepare_labels_payload = {}
        self.prepare_article_ep = ""
        self.prepare_article_payload = {}

    def clear(self):
        self.clusters_objects.clear()
        self.clusterer.dataset.clear()

    def run(
        self,
        generate_labels: bool,
        generate_articles: bool,
        check_spelling: bool = True,
    ):
        assert self.clusterer is not None
        self.prepared_clusters = self.clusterer.run()

        if not self.prepared_clusters:
            raise Exception("Clustering failed! self.clusterer.run() returned False")

        self.__prepare_clusters()
        self.__prepare_clusters_stats()

        if generate_labels:
            self.generate_labels()

        if generate_articles:
            self.generate_articles(check_spelling=check_spelling)

    def to_db_objects(
        self, day_to_summary: datetime.date, store_to_db: bool = False
    ) -> (SingleDaySummary, list[ClusterDB]):
        clustering = self.__clustering_to_db_object(store_to_db=store_to_db)

        all_db_clusters = self.__clusters_to_db_objects(
            clustering=clustering, store_to_db=store_to_db
        )

        sds = self.__single_day_summary_to_db(
            clustering=clustering,
            day_to_summary=day_to_summary,
            store_to_db=store_to_db,
        )

        return sds, all_db_clusters

    def __clustering_to_db_object(self, store_to_db: bool):
        clustering = Clustering(
            genai_labels_model=self.config.labeller_config_dict.get(
                "prepare_labels", {}
            )
            .get("config", {})
            .get("model_name")
            or "- not given -",
            genai_labels_prompt=ClusteringPrompts.GENERATE_LABEL.strip(),
            genai_article_model=self.config.labeller_config_dict.get(
                "prepare_article", {}
            )
            .get("config", {})
            .get("model_name")
            or "- not given -",
            genai_article_prompt=ClusteringPrompts.PREPARE_ARTICLE.strip(),
            clustering_options=self.clusterer.clusterer.params,
            clustering_method=self.clusterer.method,
            reducer_method=self.clusterer.reduction,
            reducer_optimizer=self.clusterer.reducer_optim,
            reducer_similarity=self.config.reducer_sim_metric,
        )

        if store_to_db:
            clustering.save()

        return clustering

    def __clusters_to_db_objects(self, clustering: Clustering, store_to_db: bool):
        all_db_clusters = []
        for cluster in self.clusters_objects.values():
            db_cluster = cluster.to_cluster_db(
                to_db=store_to_db, clustering=clustering
            )
            all_db_clusters.append(db_cluster)
        return all_db_clusters

    @staticmethod
    def __single_day_summary_to_db(
        clustering: Clustering, day_to_summary: datetime.date, store_to_db: bool
    ):
        summary = SingleDaySummary(
            day_to_summary=day_to_summary,
            when_generated=timezone.now(),
            clustering=clustering,
        )
        if store_to_db:
            summary.save()

        return summary

    def generate_labels(self):
        if not self.prepared_clusters or not len(self.clusters_objects):
            return

        self.__prepare_labeller_config()

        with tqdm(
            total=len(self.clusters_objects), desc="Labels generation"
        ) as pbar:
            for cluster in self.clusters_objects.values():
                cluster.generate_label_with_genai(
                    host=self.llms_service_host,
                    ep=self.prepare_labels_ep,
                    payload=self.prepare_labels_payload,
                )
                pbar.update(1)

    def generate_articles(self, check_spelling: bool = True):
        if not self.prepared_clusters or not len(self.clusters_objects):
            return

        self.__prepare_labeller_config()

        with tqdm(
            total=len(self.clusters_objects), desc="Creating summary of articles"
        ) as pbar:
            for cluster in self.clusters_objects.values():
                cluster.generate_article(
                    host=self.llms_service_host,
                    ep=self.prepare_article_ep,
                    payload=self.prepare_article_payload,
                    check_spelling=check_spelling,
                )
                pbar.update(1)

    def __prepare_clusters(self):
        assert len(self.clusterer.dataset.dataset) == len(
            self.clusterer.dataset.metadata
        )
        assert len(self.clusterer.dataset.dataset) == len(
            self.clusterer.clusterer.labels
        )

        for label, text, metadata in zip(
            self.clusterer.clusterer.labels,
            self.clusterer.dataset.dataset,
            self.clusterer.dataset.metadata,
        ):
            if label not in self.clusters_objects:
                self.clusters_objects[label] = Cluster(label=label)

            text_str_proper = text[: self.MAX_TEXT_CHARS_LENGTH]
            self.clusters_objects[label].append(
                text=text, proper_text=text_str_proper, metadata=metadata
            )

        for cluster in self.clusters_objects.values():
            cluster.rand_texts(num_of_texts=self.MAX_TEXTS_TO_API)

    def __prepare_clusters_stats(self):
        for cluster in self.clusters_objects.values():
            stats = cluster.prepare_stats()
            import json

            print(json.dumps(stats, indent=2, ensure_ascii=False))

    def __prepare_labeller_config(self):
        assert self.config.labeller_config_dict is not None
        assert len(self.config.labeller_config_dict)

        self.llms_service_host = self.config.labeller_config_dict[
            "llama_service_host"
        ]
        self.prepare_labels_ep = self.config.labeller_config_dict["prepare_labels"][
            "ep"
        ]
        self.prepare_labels_payload = self.config.labeller_config_dict[
            "prepare_labels"
        ]["config"]
        self.prepare_article_ep = self.config.labeller_config_dict[
            "prepare_article"
        ]["ep"]
        self.prepare_article_payload = self.config.labeller_config_dict[
            "prepare_article"
        ]["config"]


class ClusteringSimilarityController:
    MIN_MOST_SIM_CLUSTERS_COUNT = 4
    MIN_MOST_SIM_CLUSTERS_BIAS = 0.00001

    def __init__(
        self,
        clustering_config_path: str,
        device: str,
        store_to_db: bool,
        n_most_sim_clusters: int = -1,
        similarity_bias: float = None,
    ):
        self.device = device
        self.store_to_db = store_to_db

        self.similarity_metric = "cosine"

        self.similarity_bias = similarity_bias
        self.n_most_sim_clusters = n_most_sim_clusters

        self.logger = get_logger()

        self.config = ClustererConfig(config_file_path=clustering_config_path)
        self.embedder_handler = EmbeddingsDatasetHandler(
            embedder_path=self.config.embedder_path,
            embedder_input_size=self.config.embedder_input_size,
            device=self.device,
            load_model=True,
        )

    @staticmethod
    def get_clusters_without_similarity():
        return ClusterDB.objects.filter(
            Q(has_next_similarity=False) | Q(has_prev_similarity=False)
        )

    def check_similarity_of_cluster(
        self,
        cluster: ClusterDB,
        n_most_sim_clusters: int,
        similarity_bias: float,
        check_next: bool = True,
        check_prev: bool = True,
    ):
        if n_most_sim_clusters > 0:
            self.n_most_sim_clusters = n_most_sim_clusters
        assert self.n_most_sim_clusters > self.MIN_MOST_SIM_CLUSTERS_COUNT

        if similarity_bias > -0.0000001:
            self.similarity_bias = similarity_bias
        assert self.similarity_bias > self.MIN_MOST_SIM_CLUSTERS_BIAS

        cluster_day = self.__get_date_of_cluster(cluster=cluster)
        if cluster_day is None:
            raise Exception(f"Cannot resolve date for cluster {cluster.pk}")

        similarities = {}
        if check_next and not cluster.has_next_similarity:
            similarities["next"] = self.__check_similarity_of_cluster_next(
                cluster=cluster, day=cluster_day
            )

        if check_prev and not cluster.has_prev_similarity:
            similarities["prev"] = self.__check_similarity_of_cluster_prev(
                cluster=cluster, day=cluster_day
            )
        return similarities

    def __check_similarity_of_cluster_next(
        self, cluster: ClusterDB, day: datetime.date
    ):
        most_sim_clusters_in_propositions = self.__most_sim_clusters_on_the_day(
            date_for_check=day + datetime.timedelta(days=1), cluster=cluster
        )
        if self.store_to_db and most_sim_clusters_in_propositions is not None:
            ClusterDB.objects.filter(pk=cluster.pk).update(has_next_similarity=True)
        elif most_sim_clusters_in_propositions is not None:
            cluster.has_next_similarity = True

        if most_sim_clusters_in_propositions is None:
            most_sim_clusters_in_propositions = []

        return most_sim_clusters_in_propositions

    def __check_similarity_of_cluster_prev(
        self, cluster: ClusterDB, day: datetime.date
    ):
        most_sim_clusters_in_propositions = self.__most_sim_clusters_on_the_day(
            date_for_check=day + datetime.timedelta(days=-1), cluster=cluster
        )
        if self.store_to_db and most_sim_clusters_in_propositions is not None:
            ClusterDB.objects.filter(pk=cluster.pk).update(has_prev_similarity=True)
        elif most_sim_clusters_in_propositions is not None:
            cluster.has_prev_similarity = True

        if most_sim_clusters_in_propositions is None:
            most_sim_clusters_in_propositions = []

        return most_sim_clusters_in_propositions

    def __most_sim_clusters_on_the_day(
        self, date_for_check, cluster
    ) -> list or None:
        """
        Returns None if SingleDaySummary does not exist for a given date.
        Otherwise, returns a list of propositions from the
        list of SingleDaySummary clustering.

        :param date_for_check: Date for check similarity (return from database)
        :param cluster: Cluster to be checked
        :return: None or list of most similar propositions
        """
        # For a single day maybe prepared many propositions of clusters
        propositions = self.__get_clusters_for_day(day=date_for_check)
        if not len(propositions):
            return None

        # Calculate similarity to top_n most similar cluster in each proposition
        most_sim_clusters_in_propositions = []
        for clusters_prop in propositions:
            most_sim_clusters, sim_values = self.__get_n_most_similar_clusters(
                cluster=cluster,
                clusters_to_sim=clusters_prop,
                top_n=self.n_most_sim_clusters,
                similarity_bias=self.similarity_bias,
            )

            sim_clusters = self.__cluster_similarity_to_sim_results_db(
                source=cluster, targets=most_sim_clusters, sim_values=sim_values
            )

            most_sim_clusters_in_propositions.append(sim_clusters)
        return most_sim_clusters_in_propositions

    def __get_n_most_similar_clusters(
        self,
        cluster: ClusterDB,
        clusters_to_sim: list[ClusterDB],
        top_n: int,
        similarity_bias: float,
    ) -> (list[ClusterDB], list[float]):
        text = cluster.article_text
        sim_texts_str = [c.article_text for c in clusters_to_sim]

        # Prepare texts to convert to embeddings
        all_texts = [text] + sim_texts_str
        proper_all_texts = self.embedder_handler.convert_to_proper_dataset(
            texts=all_texts, show_progress=False
        )
        # Prepare embeddings of cluster and clusters_to_sim
        embeddings = self.embedder_handler.convert_to_embeddings(
            texts=proper_all_texts
        )
        assert len(proper_all_texts) == len(embeddings)

        # First elem of `embeddings` is cluster embedding
        cluster_embedding = embeddings[0]
        # the rest are embeddings of clusters_to_sim
        clusters_to_sim_embeddings = embeddings[1:]
        assert len(clusters_to_sim_embeddings) == len(sim_texts_str)

        # Calculate cosine between cluster_embedding and the clusters_to_sim_embeddings
        sim_values = []
        most_sim_clusters = []
        similarities = self.embedder_handler.model.similarity(
            numpy.array(cluster_embedding), numpy.array(clusters_to_sim_embeddings)
        )[0]

        if len(similarities) == 0:
            return [], []

        if len(similarities) < top_n:
            top_n = len(similarities)

        scores, indices = torch.topk(similarities, k=top_n)
        # Get top_n most similar clusters and its values
        for score, idx in zip(scores, indices):
            if score < similarity_bias:
                continue
            most_sim_clusters.append(clusters_to_sim[idx])
            sim_values.append(score)

        return most_sim_clusters, sim_values

    def __cluster_similarity_to_sim_results_db(
        self, source: ClusterDB, targets: list[ClusterDB], sim_values: list[float]
    ) -> list[SimilarClusters]:
        assert len(targets) == len(sim_values)

        sim_clusters = []
        clustering = source.clustering
        for target, sim_value in zip(targets, sim_values):
            self.logger.info(
                f"sim(source {source.pk},  target {target.pk}) = {sim_value}"
            )
            if SimilarClusters.objects.filter(
                source=source, target=target, clustering=clustering
            ).exists():
                sc = SimilarClusters.objects.get(
                    source=source, target=target, clustering=clustering
                )
            else:
                sc = SimilarClusters(
                    source=source,
                    target=target,
                    similarity_value=sim_value,
                    similarity_metric=self.similarity_metric,
                    similarity_model=self.config.embedder_path,
                    is_active=True,
                    clustering=clustering,
                )

                if self.store_to_db:
                    sc.save()

            sim_clusters.append(sc)
        return sim_clusters

    @staticmethod
    def __get_date_of_cluster(cluster: ClusterDB) -> datetime.date or None:
        day_to_summary = None
        for sds in SingleDaySummary.objects.filter(clustering=cluster.clustering.pk):
            day_to_summary = sds.day_to_summary
            if day_to_summary is not None:
                break
        return day_to_summary

    @staticmethod
    def __get_clusters_for_day(day: datetime.date) -> list[list]:
        clusters = []
        for sds in SingleDaySummary.objects.filter(day_to_summary=day):
            cl_list = list(ClusterDB.objects.filter(clustering=sds.clustering))
            if len(cl_list):
                clusters.append(cl_list)
        return clusters
