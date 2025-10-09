import os
from typing import Optional

import tqdm
import random
import pickle
import django
import argparse

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from system.controllers import SystemController
from creator.models import ContinuousInformationGraph
from creator.controllers.graph import AvgEmbeddingContinuousSubGraphsGeneralizer


from main.src.constants import get_logger


def prepare_parser(desc=""):
    p = argparse.ArgumentParser(description=desc)

    p.add_argument("--out-dir", dest="out_dir", required=True, type=str)
    p.add_argument(
        "--sub-graphs-dir", dest="sub_graphs_dir", required=True, type=str
    )
    p.add_argument("--device", dest="device", default="gpu", type=str)
    p.add_argument(
        "--clustering-config",
        dest="clustering_config_path",
        default="configs/clusterer-config.json",
        type=str,
    )
    p.add_argument(
        "--load-from-out-dir", dest="load_from_out_dir", action="store_true"
    )

    p.add_argument("--skip-outliers", dest="skip_outliers", action="store_true")

    return p


def get_last_active_cont_info_graph(logger) -> Optional[ContinuousInformationGraph]:
    logger.info(f"Getting last active continuous information graph...")

    a_cigs = ContinuousInformationGraph.objects.filter(is_active=True).order_by(
        "-when_created"
    )
    if not len(a_cigs):
        return None
    return a_cigs[0]


def store_to_dir(sim_hyper_graph, out_dir):
    n_d = [n for n in sim_hyper_graph.nodes(data=True)]
    random.shuffle(n_d)
    label = (
        n_d[0][1]["data"]["label"]
        .replace(" ", "_")
        .replace(".", "")
        .replace(",", "_")
        .replace(":", "_")
    )
    hg_label_str = os.path.join(out_dir, label)
    hg_label_str += "_graph.pkl"
    with open(hg_label_str, "wb") as f:
        pickle.dump(sim_hyper_graph, f)


def main(argv=None):
    args = prepare_parser(argv).parse_args(argv)

    logger = get_logger()

    # system_settings = SystemController.get_system_settings()
    # if system_settings.doing_news_information_sub_graph:
    #     return
    #
    # SystemController.begin_doing_information_sub_graph(system_settings)

    logger.info(f"Loading clustering config: {args.clustering_config_path}")
    generalizer = AvgEmbeddingContinuousSubGraphsGeneralizer(
        clustering_config_path=args.clustering_config_path,
        out_dir=args.out_dir,
        device=args.device,
    )

    last_cig = get_last_active_cont_info_graph(logger=logger)
    if args.load_from_out_dir:
        logger.info(f"Loading embeddings...")
        embeddings = generalizer.load_embeddings(out_dir=args.out_dir)
    else:
        logger.info(f"Generating embeddings...")
        embeddings = generalizer.generalize(
            cig=last_cig, skip_outliers=args.skip_outliers
        )

    sim_graphs = generalizer.calculate_similarity(
        cig=last_cig,
        sg_avg_embeddings=embeddings,
        sub_graphs_dir=args.sub_graphs_dir,
    )

    out_sim_graphs = os.path.join(args.out_dir, "hyper_graphs")
    os.makedirs(out_sim_graphs, exist_ok=True)

    with tqdm.tqdm(
        total=len(sim_graphs), desc=f"Storing hyper graphs to {out_sim_graphs}"
    ) as pbar:
        for s_g in sim_graphs:
            store_to_dir(sim_hyper_graph=s_g, out_dir=out_sim_graphs)
            pbar.update(1)

    # SystemController.end_doing_information_sub_graph(system_settings)


if __name__ == "__main__":
    main()
