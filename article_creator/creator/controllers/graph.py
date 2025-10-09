import os
import abc
import time

import tqdm
import json
import numpy
import queue
import torch
import random
import pickle
import datetime
import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_numpy

from clusterer.clustering.config import ClustererConfig
from clusterer.dataset.embedding_dataset import EmbeddingsDatasetHandler

from main.src.constants import get_logger
from creator.models import (
    GeneratedNews,
    SimilarClusters,
    Cluster,
    SingleDaySummary,
    ContinuousInformationGraph,
    ContinuousInformationSubGraph,
    ClustersInContinuousInformationSubGraph,
)


class BaseGraphI(abc.ABC):
    CACHE_DIR = None
    NODES_CACHE_FILE = None
    EDGES_CACHE_FILE = None

    def __init__(self, out_dir: str or None = None):
        self._out_dir = (
            out_dir if out_dir is not None and len(out_dir.strip()) else None
        )

        self._nodes = None
        self._edges = None

        self.is_built = False
        self.is_normalized = False
        self.is_standardized = False

        self.normalize_minmax = False
        self.normalize_tangens = False

        self.when_built = None

        assert self.CACHE_DIR is not None
        assert self.NODES_CACHE_FILE is not None
        assert self.EDGES_CACHE_FILE is not None

        self.logger = get_logger()

    @abc.abstractmethod
    def add_node(self, node_pk, node_data):
        raise NotImplemented

    @abc.abstractmethod
    def add_edge(self, source_node_pk, target_node_pk, edge_data, edge_pk=None):
        raise NotImplemented

    @abc.abstractmethod
    def build(self, normalize: bool = True, standardize: bool = False) -> bool:
        raise NotImplemented

    def summary(self) -> dict:
        nodes = [] if self._nodes is None else self._nodes
        edges = [] if self._edges is None else self._edges
        return self._summary_of_nodes_edges(nodes=nodes, edges=edges)

    def _summary_of_nodes_edges(self, nodes: list, edges: list) -> dict:
        when_built_str = ""
        if self.is_built and self.when_built is not None:
            when_built_str = self.when_built.strftime("%Y.%m.%d %H:%M:%S")

        edges_count = len(edges)
        nodes_count = len(nodes)
        density = edges_count / nodes_count if nodes_count > 0 else 0.0

        summary = {
            "class_name": self.__class__.__name__,
            "edges_count": edges_count,
            "nodes_count": nodes_count,
            "density": density,
            "when_built": when_built_str,
            "is_standardized": self.is_standardized,
            "is_normalized_tangens": self.normalize_tangens,
            "is_normalized_minmax": self.normalize_minmax,
        }
        return summary

    def set_normalization(self, normalize_tangens: bool, normalize_minmax: bool):
        self.normalize_minmax = normalize_minmax
        self.normalize_tangens = normalize_tangens

    def save_data_to_cache(
        self,
        data,
        out_dir: str,
        cache_out_dir: str,
        cache_file: str,
        data_type_str: str or None = None,
    ):
        if data_type_str is None:
            data_type_str = "default_type"

        cache_dir = os.path.join(out_dir, cache_out_dir)
        cache_file_path = os.path.join(cache_dir, cache_file)
        os.makedirs(cache_dir, exist_ok=True)
        try:
            with open(cache_file_path, "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info(f"{data_type_str} saved to cache: {cache_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save {data_type_str} to cache: {e}")
            raise

    def save(self):
        self.save_to_cache()

    def load(self):
        self.load_from_cache()

    def save_to_cache(self):
        self.save_nodes_to_cache()
        self.save_edges_to_cache()

    def load_from_cache(self):
        self.load_nodes_from_cache()
        if self._nodes is None or not len(self._nodes):
            self.is_built = False
            return

        self.load_edges_from_cache()
        if self._edges is None or not len(self._edges):
            self.is_built = False

    def save_nodes_to_cache(self):
        assert self._out_dir is not None

        self.save_data_to_cache(
            data=self._nodes,
            out_dir=self._out_dir,
            cache_out_dir=self.CACHE_DIR,
            cache_file=self.NODES_CACHE_FILE,
            data_type_str="nodes",
        )

    def save_edges_to_cache(self):
        assert self._out_dir is not None
        self.save_data_to_cache(
            data=self._edges,
            out_dir=self._out_dir,
            cache_out_dir=self.CACHE_DIR,
            cache_file=self.EDGES_CACHE_FILE,
            data_type_str="edges",
        )

    def load_nodes_from_cache(self):
        self._nodes = self.load_data_from_cache(
            cache_file=self.NODES_CACHE_FILE, data_type_str="nodes"
        )

    def load_edges_from_cache(self):
        self._edges = self.load_data_from_cache(
            cache_file=self.EDGES_CACHE_FILE, data_type_str="edges"
        )

    def load_data_from_cache(self, cache_file, data_type_str: str or None = None):
        assert self._out_dir is not None

        if data_type_str is None:
            data_type_str = "default_type"

        cache_file_path = os.path.join(self._out_dir, self.CACHE_DIR, cache_file)
        try:
            with open(cache_file_path, "rb") as handle:
                data = pickle.load(handle)
            self.logger.info(f"{data_type_str} loaded from cache: {cache_file_path}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load {data_type_str} from cache: {e}")
        return {}


class CompiledGraphI(BaseGraphI, abc.ABC):
    BASE_GRAPH_FILE = None

    """
    Compiled version of the graph.

    Should be implemented with a specific graph library dependency
    (for example, graph-tool dependency). Must be very efficient.
    """

    def __init__(self, out_dir):
        super().__init__(out_dir=out_dir)
        self.base_graph = None

        assert self.BASE_GRAPH_FILE is not None

    @abc.abstractmethod
    def save_base_graph(self, tan_as_weights: bool, minmax_as_weights: bool):
        pass

    @abc.abstractmethod
    def load_base_graph(self):
        pass


class CompiledGraphNX(CompiledGraphI):
    CACHE_DIR = "NX"
    NODES_CACHE_FILE = "CompiledGraphNX-nodes.pkl"
    EDGES_CACHE_FILE = "CompiledGraphNX-edges.pkl"
    BASE_GRAPH_FILE = "CompiledGraphNX-base_graph.pkl"
    BASE_GRAPH_FILE_INFO = "CompiledGraphNX-base_graph-info.json"

    MIN_VALUE_BIAS = 0.3

    def __init__(self, out_dir: str):
        super().__init__(out_dir=out_dir)
        self.base_graph = nx.MultiDiGraph(main=True)

        self.base_graph_dir = None
        self.base_graph_file_name = None

    def prepare_personalization(
        self, nodes: list or None
    ) -> nx.MultiDiGraph or None:
        pr_vec = {}
        const = len(nodes)
        for n in self.base_graph.nodes():
            pr_vec[n] = 0.0
            if n in nodes:
                out_deg = max(len(self.base_graph.out_edges(n)), 1)
                pr_vec[n] = 1 / (const * out_deg)

        pr_values = _pagerank_numpy(
            G=self.base_graph, personalization=pr_vec, alpha=0.85
        )

        return pr_values

    def add_node(self, node_pk, node_data):
        if self.base_graph.has_node(node_pk):
            return

        self.base_graph.add_node(node_pk, data=node_data, weight=1.0)

    def add_edge(
        self,
        source_node_pk,
        target_node_pk,
        edge_data,
        edge_pk=None,
        weight: float = 1.0,
    ):
        if self.base_graph.has_edge(source_node_pk, target_node_pk):
            return

        self.base_graph.add_edge(
            source_node_pk, target_node_pk, data=edge_data, weight=weight
        )

    def build(
        self,
        normalize: bool = True,
        standardize: bool = False,
        tan_as_weights: bool = False,
        minmax_as_weights: bool = False,
    ) -> bool:
        self.is_built = False
        self.is_normalized = False
        self.is_standardized = False

        if not self.base_graph.number_of_nodes():
            self.logger.error("CompiledGraphNX: No nodes found in graph G")
            return self.is_built

        if not self.base_graph.number_of_edges():
            self.logger.error("CompiledGraphNX: No edges found in graph G")
            return self.is_built

        self.is_built = True

        if normalize:
            self.__normalize_edges_weights(
                tan_as_weights=tan_as_weights, minmax_as_weights=minmax_as_weights
            )
            self.is_normalized = True

        if standardize:
            self.__standardize_edges_weights()
            self.is_standardized = True

        return self.is_built

    def summary(self):
        nodes = []
        edges = []
        if self.base_graph is not None:
            nodes = [] if self.base_graph.nodes is None else self.base_graph.nodes
            edges = [] if self.base_graph.edges is None else self.base_graph.edges

        return self._summary_of_nodes_edges(nodes=nodes, edges=edges)

    def save_base_graph(self, tan_as_weights: bool, minmax_as_weights: bool):
        assert self._out_dir is not None
        assert self.base_graph is not None

        self.base_graph.info = {
            "is_normalized": self.is_normalized,
            "is_standardized": self.is_standardized,
            "minmax_normalized": self.normalize_minmax,
            "tangens_normalized": self.normalize_tangens,
            "tangens_as_weights": tan_as_weights,
            "minmax_as_weights": minmax_as_weights,
            "normalization_order": "minmax(tan(sim(x, y)))",
        }

        self.base_graph_dir = os.path.join(self._out_dir, self.CACHE_DIR)
        self.base_graph_file_name = self.BASE_GRAPH_FILE

        self.save_data_to_cache(
            data=self.base_graph,
            out_dir=self._out_dir,
            cache_out_dir=self.CACHE_DIR,
            cache_file=self.BASE_GRAPH_FILE,
            data_type_str="base_graph",
        )

        summary_json_file = os.path.join(
            self._out_dir, self.CACHE_DIR, self.BASE_GRAPH_FILE_INFO
        )
        with open(summary_json_file, "wt") as handle:
            json.dump(self.summary(), handle, indent=2, ensure_ascii=False)

    def load_base_graph(self):
        assert self._out_dir is not None

        base_graph = self.load_data_from_cache(
            cache_file=self.BASE_GRAPH_FILE, data_type_str="base_graph"
        )
        if base_graph is not None and len(base_graph):
            self.base_graph = base_graph
            self.is_built = True

    def __standardize_edges_weights(self):
        all_nodes = self.base_graph.nodes(data=False)
        with tqdm.tqdm(total=len(all_nodes), desc="Standardization weights") as pbar:
            for n in all_nodes:
                self.__standardize_node_successors_weights(n=n)
                pbar.update(1)

    def __normalize_edges_weights(
        self, tan_as_weights: bool, minmax_as_weights: bool
    ):
        all_nodes = list(self.base_graph.nodes(data=False))
        with tqdm.tqdm(total=len(all_nodes), desc="Normalizing weights") as pbar:
            for n in all_nodes:
                if self.normalize_tangens:
                    self.__normalize_node_successors_weights_tangens(
                        n=n, set_as_weights=tan_as_weights
                    )

                if self.normalize_minmax:
                    self.__normalize_node_successors_weights_minmax(
                        n=n, set_as_weights=minmax_as_weights, raw_data=False
                    )
                pbar.update(1)

    def __normalize_node_successors_weights_tangens(self, n, set_as_weights: bool):
        successors = [s for s in self.base_graph.successors(n)]
        if not len(successors):
            return

        for s in successors:
            x = self.base_graph[n][s][0]["weight"]
            tan_x = float(numpy.tan(x))
            self.base_graph[n][s][0]["data"]["similarity_tangens"] = tan_x
            if set_as_weights:
                self.base_graph[n][s][0]["weight"] = tan_x

    def __normalize_node_successors_weights_minmax(
        self, n, set_as_weights: bool, raw_data: bool = False
    ):
        successors = [s for s in self.base_graph.successors(n)]
        if not len(successors):
            return

        weights = [self.base_graph[n][s][0]["weight"] for s in successors]
        if raw_data:
            min_v = min(weights)
            max_v = max(weights)
        else:
            min_v = min(weights + [self.MIN_VALUE_BIAS])
            max_v = max(weights + [min_v])

        min_max_val = max_v - min_v
        if min_max_val > 0:
            for s in successors:
                x = self.base_graph[n][s][0]["weight"]
                minmax = (x - min_v) / (max_v - min_v)
                self.base_graph[n][s][0]["data"]["similarity_minmax"] = minmax
                if set_as_weights:
                    self.base_graph[n][s][0]["weight"] = minmax

    def __standardize_node_successors_weights(self, n):
        """
        Standard scaling:

        z = (x - u) / s

        where:
            u is the mean of the training samples
            s is the standard deviation of the training samples

        :param n: Node from self.g
        :return: None
        """
        successors = [s for s in self.base_graph.successors(n)]
        if len(successors) < 2:
            return
        weights = [self.base_graph[n][s][0]["weight"] for s in successors]

        std = float(numpy.std(weights))
        mean = float(numpy.mean(weights))
        for s in successors:
            x = self.base_graph[n][s][0]["weight"]
            self.base_graph[n][s][0]["weight"] = (x - mean) / std


# ==================================================================================


class InformationClusterGraph(BaseGraphI):
    """
    Contains "graph" build on the data -- maps data to proper nodes,
    This graph always contains a compiled version -- to graph manipulation.
    """

    # InformationClusterGraph:
    CACHE_DIR = "ICG"
    NODES_CACHE_FILE = "InformationClusterGraph-nodes.pkl"
    EDGES_CACHE_FILE = "InformationClusterGraph-edges.pkl"

    # Personalization directory
    PERS_CACHE_DIR = "PERS"
    PERS_CACHE_DIR_DAYS = "days"
    PERS_NODES_PR_PERS_WEIGHTS = "pr_personalization.pkl"
    PERS_NODES_PR_PERS_WEIGHTS_SFT = "pr_personalization_sft.pkl"
    PERS_NODES_PR_PERS_WEIGHTS_MINMAX = "pr_personalization_minmax.pkl"
    PERS_NODES_PR_PERS_WEIGHTS_MM_SFT = "pr_personalization_minmax_softmax.pkl"
    # PERS_PERSONALIZED_GRAPH = "pr_personalized_g.pkl"

    def __init__(
        self,
        out_dir: str or None = None,
        use_cache: bool = False,
        try_to_load: bool = False,
    ):
        super().__init__(out_dir=out_dir)

        self._nodes = {}
        self._edges = {}
        self._use_cache = use_cache
        self._try_to_load = try_to_load

        self._personalizations_for_g = {}
        self._personalizations_for_g_softmax = {}
        self._personalizations_for_g_min_max = {}
        self._personalizations_for_g_min_max_softmax = {}

        self.compiled_graph = CompiledGraphNX(out_dir=self._out_dir)
        if self._use_cache and self._try_to_load:
            self.load()
            self.compiled_graph.load()
            self.compiled_graph.load_base_graph()

            self.is_built = self.compiled_graph.is_built

    def add_node(self, node_pk, node_data):
        if node_pk in self._nodes:
            return
        self._nodes[node_pk] = node_data

    def add_edge(self, source_node_pk, target_node_pk, edge_data, edge_pk=None):
        if source_node_pk not in self._nodes:
            self.logger.warning(f"Source node {source_node_pk} not in graph")
            return
        if target_node_pk not in self._nodes:
            self.logger.warning(f"Target node {target_node_pk} not in graph")
            return
        if edge_pk is None:
            self.logger.error(
                f"Edge identifier is required in InformationClusterGraph"
            )
            self.logger.error(
                f"Mostly creator.models.SimilarClusters objects "
                f"primary kay may be used as identifier."
            )
            return
        if edge_pk in self._edges:
            return

        if edge_pk is not None:
            self._edges[edge_pk] = {
                "source": source_node_pk,
                "target": target_node_pk,
                "data": edge_data,
            }

    def build(
        self,
        normalize: bool = True,
        standardize: bool = False,
        tan_as_weights: bool = False,
        minmax_as_weights: bool = False,
    ):
        self.logger.info(f"CompiledGraphNX len(self._nodes)={len(self._nodes)}")
        self.logger.info(f"CompiledGraphNX len(self._edges)={len(self._edges)}")

        if self._use_cache:
            self.save_nodes_to_cache()
            self.save_edges_to_cache()

        with tqdm.tqdm(total=len(self._nodes), desc="Compiling nodes") as pbar:
            # if nodes not exists, then prepare and store to out_dir cache
            for node_pk, node_data in self._nodes.items():
                self.compiled_graph.add_node(node_pk=node_pk, node_data=node_data)
                pbar.update(1)

            if self._use_cache:
                self.compiled_graph._nodes = self.compiled_graph.base_graph.nodes(
                    data=True
                )
                self.compiled_graph.save_nodes_to_cache()

        with tqdm.tqdm(total=len(self._edges), desc="Compiling edges") as pbar:
            for edge_pk, edge_data in self._edges.items():
                edge_data["edge_pk"] = edge_pk
                self.compiled_graph.add_edge(
                    source_node_pk=edge_data["source"],
                    target_node_pk=edge_data["target"],
                    edge_data=edge_data["data"],
                    weight=edge_data["data"]["similarity_value"],
                )
                pbar.update(1)

            if self._use_cache:
                self.compiled_graph._edges = self.compiled_graph.base_graph.edges(
                    data=True
                )
                self.compiled_graph.save_edges_to_cache()

        self.logger.info("Compilation complete - building graph stared...")

        if normalize:
            self.compiled_graph.set_normalization(
                normalize_tangens=self.normalize_tangens,
                normalize_minmax=self.normalize_minmax,
            )

        self.is_built = self.compiled_graph.build(
            normalize=normalize,
            standardize=standardize,
            tan_as_weights=tan_as_weights,
            minmax_as_weights=minmax_as_weights,
        )
        if self._use_cache:
            self.compiled_graph.save_base_graph(
                tan_as_weights=tan_as_weights, minmax_as_weights=minmax_as_weights
            )

        summary_str = json.dumps(
            self.compiled_graph.summary(), indent=2, ensure_ascii=False
        )
        self.logger.info(f"Summary of built graph: {summary_str}")

        self.when_built = datetime.datetime.now()

    def personalize(self, day: datetime.date, clusters: list[Cluster]):
        day_str = day.strftime("%Y%m%d")

        if day_str not in self._personalizations_for_g:
            self.__personalize(day_str=day_str, clusters=clusters)

        if day_str not in self._personalizations_for_g_softmax:
            self.__personalization_softmax(day_str=day_str)

        if day_str not in self._personalizations_for_g_min_max:
            self.__personalization_min_max_norm(day_str=day_str)

        if day_str not in self._personalizations_for_g_min_max_softmax:
            self.__personalization_min_max_softmax_norm(day_str=day_str)

    def __personalize(self, day_str, clusters: list[Cluster]):
        """
        Calculates:
            personalization(g.nodes())

        :param day_str:
        :return:
        """
        # Check if personalization for this day is already cached
        if self._use_cache:
            assert self._out_dir is not None

            personalization = self.__load_personalization_for_day(day_str=day_str)
            if personalization is not None:
                self._personalizations_for_g[day_str] = personalization
                return

        personalization = self.compiled_graph.prepare_personalization(
            nodes=[c.pk for c in clusters]
        )

        if self._use_cache and personalization is not None:
            assert self._out_dir is not None

            self.__save_personalization_for_day(
                day_str=day_str, personalization=personalization
            )

        if personalization is not None:
            self._personalizations_for_g[day_str] = personalization

    def __personalization_min_max_norm(self, day_str):
        """
        Calculates:
            minmax(personalization(g.nodes()))

        :param day_str:
        :return:
        """
        if self._use_cache:
            pers_min_max = self.__load_personalization_min_max_for_day(
                day_str=day_str
            )
            if pers_min_max is not None:
                self._personalizations_for_g[day_str] = pers_min_max
                return

        pg = self._personalizations_for_g[day_str]

        keys = [k for k in pg.keys()]
        values = numpy.array([pg[v] for v in keys], dtype=numpy.float64)

        min_v = numpy.min(values)
        max_v = numpy.max(values)
        min_max_norm = (values - min_v) / (max_v - min_v)

        pers_min_max = {}
        for k, mm in zip(keys, min_max_norm):
            pers_min_max[k] = mm

        if not len(pers_min_max):
            pers_min_max = None

        if self._use_cache and pers_min_max is not None:
            self.__save_personalization_min_max_for_day(
                day_str=day_str, pers_min_max=pers_min_max
            )

        if pers_min_max is not None:
            self._personalizations_for_g_min_max[day_str] = pers_min_max

    def __personalization_softmax(self, day_str):
        """
        Calculates:
            softmax(personalization(g.nodes()))

        :param day_str:
        :return:
        """
        if self._use_cache:
            pers_sft = self.__load_personalization_sft_for_day(day_str=day_str)
            if pers_sft is not None:
                self._personalizations_for_g_softmax[day_str] = pers_sft
                return

        pg = self._personalizations_for_g[day_str]
        keys = [k for k in pg.keys()]
        values = numpy.array([pg[v] for v in keys], dtype=numpy.float64)

        sft_max = self.__calculate_softmax(
            x=values, normalize=True, temperature=0.5, is_normalized=False
        )

        pers_sft = {}
        for k, sft in zip(keys, sft_max):
            pers_sft[k] = sft

        if not len(pers_sft):
            pers_sft = None

        if self._use_cache and pers_sft is not None:
            self.__save_personalization_sft_for_day(
                day_str=day_str, pers_sft=pers_sft
            )

        if pers_sft is not None:
            self._personalizations_for_g_softmax[day_str] = pers_sft

    def __personalization_min_max_softmax_norm(self, day_str: str):
        """
        Calculates:
            softmax(minmax(personalization(g.nodes()))

        :param day_str:
        :return:
        """
        # Check if exists _personalizations_for_g_min_max_softmax
        if self._use_cache:
            mm_sft = self.__load_personalization_mm_sft_for_day(day_str=day_str)
            if mm_sft is not None:
                self._personalizations_for_g_min_max_softmax[day_str] = mm_sft
                return

        # Use previously calculated min-max
        d_mm = self._personalizations_for_g_min_max[day_str]
        keys = [k for k in d_mm.keys()]
        values = [d_mm[k] for k in keys]

        # Convert min-max to softmax
        mm_sft = self.__calculate_softmax(values, is_normalized=True)
        mm_sft_dict = {}
        for k, _mm_sft in zip(keys, mm_sft):
            mm_sft_dict[k] = _mm_sft
        if not len(mm_sft_dict):
            mm_sft_dict = None

        # save softmax(min-max)
        if self._use_cache and mm_sft_dict is not None:
            self.__save_personalization_mm_sft_for_day(
                day_str=day_str, pers_min_max_softmax=mm_sft_dict
            )

        if mm_sft_dict is not None:
            self._personalizations_for_g_min_max_softmax[day_str] = mm_sft_dict

    def __load_personalization_for_day(self, day_str: str):
        assert self._out_dir is not None

        day_cache_file = os.path.join(
            self._out_dir,
            self.PERS_CACHE_DIR,
            self.PERS_CACHE_DIR_DAYS,
            day_str,
            self.PERS_NODES_PR_PERS_WEIGHTS,
        )
        return self._load_personalization(
            cache_file_path=day_cache_file, day_str=day_str, pers_type="pagerank"
        )

    def __load_personalization_sft_for_day(self, day_str: str):
        assert self._out_dir is not None

        day_cache_file = os.path.join(
            self._out_dir,
            self.PERS_CACHE_DIR,
            self.PERS_CACHE_DIR_DAYS,
            day_str,
            self.PERS_NODES_PR_PERS_WEIGHTS_SFT,
        )
        return self._load_personalization(
            cache_file_path=day_cache_file, day_str=day_str, pers_type="softmax"
        )

    def __load_personalization_mm_sft_for_day(self, day_str: str):
        assert self._out_dir is not None

        day_cache_file = os.path.join(
            self._out_dir,
            self.PERS_CACHE_DIR,
            self.PERS_CACHE_DIR_DAYS,
            day_str,
            self.PERS_NODES_PR_PERS_WEIGHTS_MM_SFT,
        )
        return self._load_personalization(
            cache_file_path=day_cache_file,
            day_str=day_str,
            pers_type="minmax_softmax",
        )

    def __load_personalization_min_max_for_day(self, day_str: str):
        assert self._out_dir is not None

        day_cache_file = os.path.join(
            self._out_dir,
            self.PERS_CACHE_DIR,
            self.PERS_CACHE_DIR_DAYS,
            day_str,
            self.PERS_NODES_PR_PERS_WEIGHTS_MINMAX,
        )
        return self._load_personalization(
            cache_file_path=day_cache_file, day_str=day_str, pers_type="min_max"
        )

    def _load_personalization(
        self, cache_file_path: str, day_str: str, pers_type: str
    ):
        try:
            with open(cache_file_path, "rb") as handle:
                data = pickle.load(handle)
            self.logger.info(
                f"Personalization {pers_type} for day {day_str} "
                f"is loaded from cache: {cache_file_path}"
            )
            return data
        except Exception as e:
            self.logger.error(
                f"Failed to load personalization {pers_type} for day {day_str}: {e}"
            )
        return None

    def __save_personalization_for_day(self, day_str: str, personalization):
        assert self._out_dir is not None

        day_cache_dir = os.path.join(
            self._out_dir, self.PERS_CACHE_DIR, self.PERS_CACHE_DIR_DAYS, day_str
        )

        self._save_personalization(
            cache_dir=day_cache_dir,
            cache_filename=self.PERS_NODES_PR_PERS_WEIGHTS,
            data=personalization,
            day_str=day_str,
            pers_type="pagerank",
        )

    def __save_personalization_mm_sft_for_day(
        self, day_str: str, pers_min_max_softmax
    ):
        assert self._out_dir is not None

        day_cache_dir = os.path.join(
            self._out_dir, self.PERS_CACHE_DIR, self.PERS_CACHE_DIR_DAYS, day_str
        )

        self._save_personalization(
            cache_dir=day_cache_dir,
            cache_filename=self.PERS_NODES_PR_PERS_WEIGHTS_MM_SFT,
            data=pers_min_max_softmax,
            day_str=day_str,
            pers_type="pagerank",
        )

    def __save_personalization_sft_for_day(self, day_str: str, pers_sft):
        day_cache_dir = os.path.join(
            self._out_dir, self.PERS_CACHE_DIR, self.PERS_CACHE_DIR_DAYS, day_str
        )

        self._save_personalization(
            cache_dir=day_cache_dir,
            cache_filename=self.PERS_NODES_PR_PERS_WEIGHTS_SFT,
            data=pers_sft,
            day_str=day_str,
            pers_type="softmax",
        )

    def __save_personalization_min_max_for_day(self, day_str: str, pers_min_max):
        day_cache_dir = os.path.join(
            self._out_dir, self.PERS_CACHE_DIR, self.PERS_CACHE_DIR_DAYS, day_str
        )

        self._save_personalization(
            cache_dir=day_cache_dir,
            cache_filename=self.PERS_NODES_PR_PERS_WEIGHTS_MINMAX,
            data=pers_min_max,
            day_str=day_str,
            pers_type="min_max",
        )

    def _save_personalization(
        self, cache_dir: str, cache_filename: str, data, day_str: str, pers_type: str
    ):
        os.makedirs(cache_dir, exist_ok=True)

        try:
            day_cache_file = os.path.join(cache_dir, cache_filename)
            with open(day_cache_file, "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            self.logger.info(
                f"{pers_type} personalization of graph for day {day_str} "
                f"is saved to cache: {day_cache_file}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to save {pers_type} personalization "
                f"of graph for day {day_str}: {e}"
            )
            raise e

    @staticmethod
    def __calculate_softmax(
        x,
        normalize: bool = True,
        temperature: float = 1.0,
        is_normalized: bool = False,
    ):
        if type(x) is not numpy.array:
            x = numpy.array(x, dtype=numpy.float64)

        if is_normalized:
            return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=0)

        if normalize:
            mean = numpy.mean(x)
            std = numpy.std(x)
            x = (x - mean) / std

        x = x / temperature
        x_shifted = x - numpy.max(x)
        exps = numpy.exp(x_shifted)

        return exps / numpy.sum(exps)


# ==================================================================================


class InformationGraphController:
    DEFAULT_WORKDIR = "./workdir"
    DEFAULT_DAY_2_DAY_WEIGHT = 0.5

    def __init__(
        self,
        out_dir: str or None = None,
        use_cache: bool = False,
        try_to_load: bool = False,
    ):
        self.out_dir = out_dir
        self.use_cache = use_cache
        self.try_to_load = try_to_load

        self.is_loaded = False
        self.is_personalized = False
        self.has_day_to_day_edges = False

        # Object of info graph (db)
        self.db_cont_info_graph = None

        self._clusters_objects = {}
        self.logger = get_logger()

        if use_cache:
            if out_dir is None or not len(out_dir):
                workdir_date = datetime.datetime.now().strftime("%Y%m%d__%H%M%S")
                self.out_dir = os.path.join(self.DEFAULT_WORKDIR, workdir_date)

                self.logger.warning("cache_steps is enabled but out_dir is None")
                self.logger.warning(
                    f"workdir is automatically set to: {self.out_dir}"
                )

        self.information_graph = InformationClusterGraph(
            use_cache=self.use_cache, out_dir=self.out_dir, try_to_load=try_to_load
        )

        if self.try_to_load:
            self.logger.info(
                f"Summary of the graph loaded from the cache: {self.summary()}"
            )
            self.is_loaded = self.information_graph.is_built

    def add_cluster_node(self, cluster: Cluster):
        if not cluster.is_active:
            self.logger.warning(
                f"Cluster {cluster.pk} is not active, will be skipped!"
            )
            return

        if cluster.pk not in self._clusters_objects:
            self._clusters_objects[cluster.pk] = cluster
        cluster = self._clusters_objects[cluster.pk]

        self.information_graph.add_node(
            node_pk=self.__cluster_as_node_pk(cluster=cluster),
            node_data=self.__cluster_as_node_data(cluster=cluster),
        )

    def add_similarity_edge(self, similarity: SimilarClusters):
        src_cluster = similarity.source
        trg_cluster = similarity.target
        self.information_graph.add_edge(
            source_node_pk=src_cluster.pk,
            target_node_pk=trg_cluster.pk,
            edge_pk=self.__cluster_sim_as_edge_pk(similarity=similarity),
            edge_data=self.__cluster_sim_as_edge_data(similarity=similarity),
        )

    def summary(self) -> dict:
        return self.information_graph.summary()

    def prepare_graph(
        self,
        normalize_tangens: bool,
        normalize_minmax: bool,
        standardize: bool,
        tan_as_weights: bool,
        minmax_as_weights: bool,
    ):
        self.db_cont_info_graph = None

        self.logger.info(
            f"Compiling InformationClusterGraph graph to "
            f"{type(self.information_graph.compiled_graph.base_graph)}"
        )
        _normalize = normalize_tangens or normalize_minmax
        if _normalize:
            self.information_graph.set_normalization(
                normalize_tangens=normalize_tangens,
                normalize_minmax=normalize_minmax,
            )
        self.information_graph.build(
            normalize=_normalize,
            standardize=standardize,
            tan_as_weights=tan_as_weights,
            minmax_as_weights=minmax_as_weights,
        )

        if self.use_cache:
            self.information_graph.save_nodes_to_cache()
            self.information_graph.save_edges_to_cache()

    def store_continuous_information_graph_to_db(
        self, deactivate_other_info_graphs: bool = True
    ) -> None or ContinuousInformationGraph:
        self.logger.info("Storing continuous information graph to database")

        # Prepare a new continuous info graph:
        self.db_cont_info_graph = self.__add_continuous_info_graph()
        if self.db_cont_info_graph is None:
            self.logger.warning("Error while building Continuous-Information-Graph")
            return None

        # When everything is OK, then deactivate other active info graphs
        if deactivate_other_info_graphs:
            self.__deactivate_continuous_info_graphs(
                instead_of=self.db_cont_info_graph
            )

        return self.db_cont_info_graph

    def add_day_to_day_edge(
        self, clusters_from: list[Cluster], clusters_to: list[Cluster]
    ):
        for c_f in clusters_from:
            for c_t in clusters_to:
                data = {"label": "day2day"}
                self.information_graph.compiled_graph.add_edge(
                    source_node_pk=c_f.pk,
                    target_node_pk=c_t.pk,
                    edge_data=data,
                    edge_pk=None,
                    weight=self.DEFAULT_DAY_2_DAY_WEIGHT,
                )

    def personalize_day(self, day: SingleDaySummary, clusters: list[Cluster]):
        self.information_graph.personalize(day=day.day_to_summary, clusters=clusters)

    def __add_continuous_info_graph(self) -> None or ContinuousInformationGraph:
        info_dict = {}
        if hasattr(self.information_graph.compiled_graph.base_graph, "info"):
            info_dict = self.information_graph.compiled_graph.base_graph.info
        g_dir = self.information_graph.compiled_graph.base_graph_dir
        g_f_name = self.information_graph.compiled_graph.base_graph_file_name
        summary_dict = self.information_graph.compiled_graph.summary()

        return ContinuousInformationGraph.objects.create(
            is_active=True,
            has_sub_graphs=False,
            graph_directory=g_dir,
            graph_file_name=g_f_name,
            info=info_dict,
            summary=summary_dict,
        )

    @staticmethod
    def __deactivate_continuous_info_graphs(
        instead_of: ContinuousInformationGraph or None,
    ):
        base_q_set = ContinuousInformationGraph.objects.filter(is_active=True)
        if instead_of is not None:
            base_q_set = base_q_set.exclude(pk=instead_of.pk)
        base_q_set.update(is_active=False)

    @staticmethod
    def __cluster_as_node_pk(cluster: Cluster):
        return cluster.pk

    @staticmethod
    def __cluster_as_node_data(cluster: Cluster):
        date_str = ""
        for sd in SingleDaySummary.objects.filter(
            clustering=cluster.clustering, is_active=True
        ):
            date_str = sd.day_to_summary.strftime("%Y.%m.%d")
            break

        return {
            "db_pk": cluster.pk,
            "label": cluster.label_str,
            "date": date_str,
            "size": len(cluster.news_urls),
            "is_outlier": cluster.is_outlier,
            "text": cluster.article_text,
            "news_urls": cluster.news_urls,
            "stats": cluster.stats,
        }

    @staticmethod
    def __cluster_sim_as_edge_data(similarity: SimilarClusters):
        return {
            "db_pk": similarity.pk,
            "similarity_metric": similarity.similarity_metric,
            "similarity_value": similarity.similarity_value,
        }

    @staticmethod
    def __cluster_sim_as_edge_pk(similarity: SimilarClusters):
        return similarity.pk


# ==================================================================================


class SubGraphsExtractor(abc.ABC):
    def __init__(self, g):
        self.g = g

        self.logger = get_logger()

    @abc.abstractmethod
    def extract(self):
        raise Exception


class BFSSubGraphsExtractor(SubGraphsExtractor):
    def __init__(self, g, min_sim: float = 0.0, min_tan: float = 0.0):
        super().__init__(g=g)

        self._min_sim = min_sim
        self._min_tan = min_tan

        self._sub_graphs = []
        self._allocated_nodes = set()

    def save_sub_graphs(self, out_dir: str):
        if not self._sub_graphs:
            self.logger.warning("No sub-graphs to save")
            return

        os.makedirs(out_dir, exist_ok=True)

        saved_count = 0
        with tqdm.tqdm(
            total=len(self._sub_graphs), desc="Saving sub-graphs..."
        ) as pbar:
            for i, sub_graph in enumerate(self._sub_graphs):
                if sub_graph is None or sub_graph.number_of_nodes() == 0:
                    # self.logger.warning(f"Sub-graph {i} is empty, skipping")
                    pbar.update(1)
                    continue

                nodes_dates = []
                sub_nodes = [n for n in sub_graph.nodes(data=True)]
                for n_data in [
                    s[1].get("data", {}).get("date", None) for s in sub_nodes
                ]:
                    if n_data is None:
                        continue
                    nodes_dates.append(
                        datetime.datetime.strptime(n_data, "%Y.%m.%d")
                    )

                _r_n = random.choice(sub_nodes)
                _r_n_data = sub_graph.nodes.get(_r_n[0], {})
                node_label = _r_n_data.get("data", {}).get("label", f"node_{_r_n}")
                proper_filename = "".join(
                    c for c in str(node_label) if c.isalnum() or c in (" ", "-", "_")
                ).rstrip()
                proper_filename = proper_filename.replace(" ", "_")
                if len(nodes_dates):
                    min_date = min(nodes_dates).strftime("%Y.%m.%d")
                    max_date = max(nodes_dates).strftime("%Y.%m.%d")
                    proper_filename = f"{min_date}-{max_date}__{proper_filename}"

                if not proper_filename:
                    proper_filename = f"subgraph_{_r_n}"

                filename = f"{proper_filename}_{i}_graph.pkl"
                filepath = os.path.join(out_dir, filename)

                sub_graph.info = {
                    "min_date": min_date,
                    "max_date": max_date,
                    "label": node_label,
                    "full_filename": filename,
                }

                try:
                    with open(filepath, "wb") as f:
                        pickle.dump(sub_graph, f, protocol=pickle.HIGHEST_PROTOCOL)

                    # self.logger.info(
                    #     f"Saved sub-graph {i} with {sub_graph.number_of_nodes()} nodes "
                    #     f"and {sub_graph.number_of_edges()} edges to: {filepath}"
                    # )
                    saved_count += 1
                except Exception as e:
                    self.logger.error(
                        f"Failed to save sub-graph {i} to {filepath}: {e}"
                    )

                pbar.update(1)

        self.logger.info(f"Successfully saved {saved_count} sub-graphs to {out_dir}")

    def store_sub_graphs_to_database(
        self,
        cont_info_graph: ContinuousInformationGraph,
        out_dir: str,
        type_str: str,
    ):
        with tqdm.tqdm(
            total=len(self._sub_graphs), desc="Storing subgraphes to database..."
        ) as pbar:
            for sub_graph in self._sub_graphs:
                _cis = self.__add_continuous_subgraph_to_database(
                    sub_graph=sub_graph,
                    cig=cont_info_graph,
                    out_dir=out_dir,
                    type_str=type_str,
                )
                self.__add_clusters_to_cont_sub_graph(sub_graph, sub_graph_db=_cis)
                pbar.update(1)

        ContinuousInformationGraph.objects.filter(pk=cont_info_graph.pk).update(
            has_sub_graphs=len(self._sub_graphs) > 0
        )

    def extract(self):
        self.logger.info(f"Extracting sub-graphs from graph {self.g}...")
        self.logger.info(f"  -> nodes count {self.g.number_of_nodes()}")
        self.logger.info(f"  -> edges count {self.g.number_of_edges()}")

        _nodes_shuffled = [n for n in self.g.nodes()]
        random.shuffle(_nodes_shuffled)

        for node in _nodes_shuffled:
            if node in self._allocated_nodes:
                continue

            sub_g = self._extract_sub_graph_begin_from_node(start_node=node)
            if sub_g is None:
                self.logger.warning(f"Failed to extract sub-graph from node {node}")

            if sub_g.number_of_nodes() < 1:
                # self.logger.warning(f"Skipped empty graph...")
                continue

            self._sub_graphs.append(sub_g)

    def _extract_sub_graph_begin_from_node(self, start_node, g=None):
        g = self.g if g is None else g
        assert g is not None

        # Initialize queue with start node
        sub_g = nx.MultiDiGraph()

        g_queue = queue.Queue()
        g_queue.put(start_node)
        while not g_queue.empty():
            start_node = g_queue.get()
            if start_node in self._allocated_nodes:
                continue

            # out edges
            for _ns, _nt, _nd in g.out_edges([start_node], data=True):
                if not self._add_to_subgraph(
                    g=g, ns=_ns, nt=_nt, nd=_nd, sub_g=sub_g
                ):
                    continue

                self._allocated_nodes.add(_ns)
                g_queue.put(_nt)

            # in edges
            for _ns, _nt, _nd in g.in_edges([start_node], data=True):
                if not self._add_to_subgraph(
                    g=g, ns=_ns, nt=_nt, nd=_nd, sub_g=sub_g
                ):
                    continue

                self._allocated_nodes.add(_nt)
                g_queue.put(_ns)

        return sub_g

    def _add_to_subgraph(self, g, ns, nt, nd, sub_g):
        _data = nd.get("data", {})
        if not len(_data):
            return False

        if not self.__is_similarity_ok(edge_data=_data):
            return False

        self.__add_local_change(ns=ns, nt=nt, nd=nd, g=g, sub_g=sub_g)

        return True

    def __is_similarity_ok(self, edge_data):
        #  -> filter by similarity
        similarity_value = edge_data.get("similarity_value", 0.0)
        if similarity_value < self._min_sim:
            return False

        #  -> filter by tangens
        similarity_tangens = edge_data.get("similarity_tangens", 0.0)
        if similarity_tangens < self._min_tan:
            return False

        return True

    def __add_continuous_subgraph_to_database(
        self,
        sub_graph: nx.MultiDiGraph,
        cig: ContinuousInformationGraph,
        out_dir: str,
        type_str: str,
    ) -> ContinuousInformationSubGraph or None:
        if hasattr(sub_graph, "info"):
            info_dict = {
                "min_date": sub_graph.info["min_date"],
                "max_date": sub_graph.info["max_date"],
            }

            cig_obj, created = ContinuousInformationSubGraph.objects.get_or_create(
                graph=cig,
                label=sub_graph.info["label"],
                label_str=sub_graph.info["full_filename"],
                sub_graph_directory=out_dir,
                sub_graph_file_name=sub_graph.info["full_filename"],
                info=info_dict,
                is_active=True,
                type_str=type_str,
            )
            return cig_obj

        self.logger.error("Subgraph does not have a info attribute")
        return None

    def __add_clusters_to_cont_sub_graph(
        self, sub_graph: nx.MultiDiGraph, sub_graph_db: ContinuousInformationSubGraph
    ):
        if sub_graph_db is None:
            self.logger.error("Continuous Info Subgraph is None. Cannot add to db!")
            return

        for node, node_data in sub_graph.nodes(data=True):
            db_pk = node_data["data"]["db_pk"]
            cl_obj = list(Cluster.objects.filter(pk=db_pk))
            if not len(cl_obj):
                self.logger.error(f"Cluster {db_pk} does not exist!")
                continue

            ClustersInContinuousInformationSubGraph.objects.get_or_create(
                sub_graph=sub_graph_db,
                cluster=cl_obj[0],
            )

    @staticmethod
    def __add_local_change(ns, nt, nd, g, sub_g):
        #  -> add source node if not exists
        if not sub_g.has_node(ns):
            sub_g.add_node(ns, data=g.nodes[ns].get("data", {}))

        #  -> add target node if not exists
        if not sub_g.has_node(nt):
            sub_g.add_node(nt, data=g.nodes[nt].get("data", {}))

        #  -> add edge (don't check if exists -- there is multidigraph)
        sub_g.add_edge(ns, nt, data=nd.get("data", None))


# ==================================================================================


class ContinuousSubGraphsGeneralizer(abc.ABC):
    def __init__(self):
        self.sub_graphs = []

        self.logger = get_logger()

    @abc.abstractmethod
    def generalize(self, cig: ContinuousInformationGraph, skip_outliers: bool):
        pass

    def get_sub_graphs(
        self, cig: ContinuousInformationGraph
    ) -> list[ContinuousInformationSubGraph]:
        self.sub_graphs = list(
            ContinuousInformationSubGraph.objects.filter(graph=cig)
        )
        return self.sub_graphs

    def get_subgraph_list_of_news(
        self, subgraph: ContinuousInformationSubGraph, skip_outliers: bool
    ) -> list[GeneratedNews]:
        news_list = []
        for cic in ClustersInContinuousInformationSubGraph.objects.filter(
            sub_graph=subgraph
        ):
            if skip_outliers and cic.cluster.is_outlier:
                continue

            for a_dict in cic.cluster.news_metadata:
                news_pk = a_dict.get("news_pk")
                if news_pk is None:
                    self.logger.error(
                        "news_pk not exists in news_metadata dictionary"
                    )
                    continue

                news = None
                try:
                    news = GeneratedNews.objects.get(pk=news_pk)
                except GeneratedNews.DoesNotExist:
                    self.logger.error(f"Cannot find news with pk: {news_pk}")

                if news is not None:
                    news_list.append(news)
        return news_list


class EmbeddingContinuousSubGraphsGeneralizer(
    ContinuousSubGraphsGeneralizer, abc.ABC
):
    MIN_SIM_VALUE_TO_SAME_GRAPH = 0.725
    MIN_SIM_TAN_VALUE_TO_SAME_GRAPH = 1.3
    AVG_EMBEDDINGS_PICKLE_FILE = "avg-embeddings.pickle"

    def __init__(self, clustering_config_path: str, out_dir: str, device: str):
        super().__init__()

        self.device = device
        self.out_dir = out_dir.strip()

        if len(self.out_dir) and not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

        self.model = None
        self.clustering_config_path = clustering_config_path

        self.clustering_config = None
        self.embedder_handler = None

        self.__load_model()

    def __load_model(self):
        self.logger.info(
            f"Loading embedder model based on clustering "
            f"config: {self.clustering_config_path}"
        )
        self.clustering_config = ClustererConfig(
            config_file_path=self.clustering_config_path
        )

        self.logger.info(f" -     device: {self.device}")
        self.logger.info(f" -      model: {self.clustering_config.embedder_path}")
        self.logger.info(
            f" - input size: {self.clustering_config.embedder_input_size}"
        )

        self.embedder_handler = EmbeddingsDatasetHandler(
            embedder_path=self.clustering_config.embedder_path,
            embedder_input_size=self.clustering_config.embedder_input_size,
            device=self.device,
            load_model=True,
        )


class AvgEmbeddingContinuousSubGraphsGeneralizer(
    EmbeddingContinuousSubGraphsGeneralizer
):
    def __init__(
        self, clustering_config_path: str, out_dir: str, device: str = "cuda"
    ):
        super().__init__(
            clustering_config_path=clustering_config_path,
            out_dir=out_dir,
            device=device,
        )

    def generalize(self, cig: ContinuousInformationGraph, skip_outliers: bool):
        sub_graphs = self.get_sub_graphs(cig=cig)

        # TODO: Do usuniecia!
        # sub_graphs = sub_graphs[:20]

        sg_map = self.__map_to_list_of_news(sub_graphs, skip_outliers=skip_outliers)

        articles_count = sum(len(v) for v in sg_map.values())
        self.logger.info(f"Number of subgraphes to merge: {len(sub_graphs)}")
        self.logger.info(f"Number of articles to merge  : {articles_count}")

        sg_avg_embeddings = self.__prepare_avg_embeddings(sg_map=sg_map)
        self.__store_embeddings_to_out_dir(sg_embeddings=sg_avg_embeddings)

        return sg_avg_embeddings

    def calculate_similarity(
        self, cig: ContinuousInformationGraph, sg_avg_embeddings, sub_graphs_dir: str
    ):
        similarities = self.__calculate_similarity(sg_embeddings=sg_avg_embeddings)

        sg_similar_embeddings = []
        total = sum(len(s) for s in similarities.values())
        with tqdm.tqdm(
            total=total, desc="Building hyper graphs from sub-graphs"
        ) as pbar:
            for sg_from, sg_to_dict in similarities.items():
                for sg_to, sim_value in sg_to_dict.items():
                    sim_value = float(numpy.tan(sim_value))
                    if sim_value > self.MIN_SIM_TAN_VALUE_TO_SAME_GRAPH:
                        sg_similar_embeddings = self._make_as_same_sub_graphs(
                            sg1=sg_from,
                            sg2=sg_to,
                            similar_list=sg_similar_embeddings,
                        )
                    pbar.update(1)

        sg_similar_embeddings = self._make_as_same_sub_graphs_unique(
            similar_list=sg_similar_embeddings
        )

        sim_graphs = []
        with tqdm.tqdm(
            total=len(sg_similar_embeddings), desc="Merging sub-graphs"
        ) as pbar:
            for sim_list in sg_similar_embeddings:
                sim_graph = self.__merge_graphs(sub_graphs_dir, sim_list)
                if sim_graph is not None and len(sim_graph):
                    sim_graphs.append(sim_graph)
                pbar.update(1)

        self.logger.info(f"Number of hyper graphs: {len(sim_graphs)}")

        return sim_graphs

    @staticmethod
    def __merge_graphs(sub_graphs_dir, sg_similar_embeddings):
        _g = nx.MultiDiGraph()
        for s in sg_similar_embeddings:
            s_path = os.path.join(sub_graphs_dir, s.label_str)
            if not os.path.exists(s_path):
                raise FileNotFoundError(f"Cannot find {s_path}")
            with open(s_path, "rb") as f:
                g = pickle.load(f)
                _g = nx.compose(g, _g)
        return _g

    def load_embeddings(self, out_dir: str or None = None):
        if out_dir is not None and len(out_dir.strip()) > 0:
            self.out_dir = out_dir.strip()

        e_path = os.path.join(out_dir, self.AVG_EMBEDDINGS_PICKLE_FILE)
        if not os.path.exists(e_path):
            raise FileNotFoundError(f"Embeddings file not found: {e_path}")

        with open(e_path, "rb") as f:
            embedding = pickle.load(f)
        return embedding

    @staticmethod
    def _make_as_same_sub_graphs(
        sg1: ContinuousInformationSubGraph,
        sg2: ContinuousInformationSubGraph,
        similar_list: list,
    ):
        found = False
        for local_sim_list in similar_list:
            if sg1 in local_sim_list:
                found = True
                local_sim_list.append(sg2)
            elif sg2 in local_sim_list:
                found = True
                local_sim_list.append(sg1)
        if not found:
            similar_list.append([sg1, sg2])
        return similar_list

    @staticmethod
    def _make_as_same_sub_graphs_unique(similar_list: list):
        uq_list = []
        for similarities in similar_list:
            _sim = []
            for s in similarities:
                if s not in _sim:
                    _sim.append(s)
            uq_list.append(_sim)
        return uq_list

    def __map_to_list_of_news(
        self, sub_graphs: list[ContinuousInformationSubGraph], skip_outliers: bool
    ):
        sg_map = {}
        with tqdm.tqdm(
            total=len(sub_graphs),
            desc="Mapping generated news to subgraphes before merging",
        ) as pbar:
            for sg in sub_graphs:
                sg_news_list = self.get_subgraph_list_of_news(
                    subgraph=sg, skip_outliers=skip_outliers
                )
                if len(sg_news_list):
                    sg_map[sg] = sg_news_list
                pbar.update(1)
        return sg_map

    def __prepare_avg_embeddings(self, sg_map: dict):
        sg_emb = {}
        for sg, list_of_news in sg_map.items():
            sg_emb[sg] = self.__avg_embeddings(
                embeddings=self.embedder_handler.convert_to_embeddings(
                    texts=[l.generated_text for l in list_of_news]
                )
            )
        return sg_emb

    def __store_embeddings_to_out_dir(self, sg_embeddings: dict):
        sg_file_name = os.path.join(self.out_dir, self.AVG_EMBEDDINGS_PICKLE_FILE)
        self.logger.info(f"Saving embeddings to file: {sg_file_name}")

        with open(sg_file_name, "wb") as fout:
            pickle.dump(sg_embeddings, fout, protocol=pickle.HIGHEST_PROTOCOL)

    def __calculate_similarity(self, sg_embeddings: dict):
        similarities = {}
        sub_graphs_ids = [k for k in sg_embeddings.keys()]
        sub_graphs_embs = [sg_embeddings[e] for e in sub_graphs_ids]

        total = int((len(sub_graphs_ids) ** 2 / 2) - len(sub_graphs_ids))
        with tqdm.tqdm(total=total, desc="Calculating similarity") as pbar:
            for i in range(0, len(sub_graphs_ids)):
                i_pk = sub_graphs_ids[i]
                i_emb = sub_graphs_embs[i]
                for j in range(i + 1, len(sub_graphs_ids)):
                    j_pk = sub_graphs_ids[j]
                    j_emb = sub_graphs_embs[j]
                    ij_sim_value = self.embedder_handler.model.similarity(
                        numpy.array(i_emb), numpy.array(j_emb)
                    )[0].item()
                    if i_pk not in similarities:
                        similarities[i_pk] = {}
                    similarities[i_pk][j_pk] = ij_sim_value
                    pbar.update(1)
        return similarities

    @staticmethod
    def __avg_embeddings(embeddings: list[torch.Tensor]):
        return torch.mean(torch.Tensor(embeddings), dim=0)
