import os
from typing import Optional

import tqdm
import pickle
import logging
import django
import argparse

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from system.controllers import SystemController
from creator.models import (
    ContinuousInformationGraph,
    ContinuousInformationSubGraph,
    ClustersInContinuousInformationSubGraph,
)
from creator.controllers.graph import BFSSubGraphsExtractor


def prepare_parser(desc=""):
    p = argparse.ArgumentParser(description=desc)

    p.add_argument(
        "-g",
        "--graph-path",
        dest="graph_path",
        required=False,
        help="Path to the graph file. If not given, "
        "then last Continuous-Information-Graph will be used.",
    )
    p.add_argument(
        "-m",
        "--min-sim-value",
        dest="min_sim_value",
        required=False,
        default=0.4,
        type=float,
    )
    p.add_argument(
        "-t",
        "--min-tg-value",
        dest="min_tg_value",
        required=False,
        default=0.4,
        type=float,
    )
    p.add_argument("--out-dir", dest="out_dir", required=True, type=str)
    p.add_argument(
        "--type", dest="type", required=False, type=str, default="default"
    )

    p.add_argument(
        "--remove-previous-subgraphes", dest="remove_previous", action="store_true"
    )

    return p


def get_last_active_cont_info_graph_without_sub_graphs() -> (
    Optional[ContinuousInformationGraph]
):
    a_cigs = ContinuousInformationGraph.objects.filter(is_active=True).order_by(
        "-when_created"
    )
    if not len(a_cigs):
        return None
    return a_cigs[0]


def delete_subgraphs(instead_of: ContinuousInformationGraph, type_str: str) -> None:
    """
    Remove:
        ContinuousInformationSubGraph
        ClustersInContinuousInformationSubGraph

    :param type_str:
    :param instead_of:
    :return:
    """
    all_con_inf_sub_graphs = list(
        ContinuousInformationSubGraph.objects.filter(type_str=type_str).exclude(
            graph__pk=instead_of.pk
        )
    )

    with tqdm.tqdm(
        total=len(all_con_inf_sub_graphs),
        desc="Deleting old subgraphs and clusters-in-subgraphs",
    ) as pbar:
        for cis in all_con_inf_sub_graphs:
            ClustersInContinuousInformationSubGraph.objects.filter(
                sub_graph=cis
            ).delete()
            cis.delete()
            pbar.update(1)


def main(argv=None):
    args = prepare_parser(argv).parse_args(argv)

    system_settings = SystemController.get_system_settings()
    if system_settings.doing_news_information_sub_graph:
        return

    SystemController.begin_doing_information_sub_graph(system_settings)

    cig = None
    if args.graph_path is None:
        cig = get_last_active_cont_info_graph_without_sub_graphs()
        if cig is None:
            SystemController.end_doing_information_graph(system_settings)
            logging.error("No active info graphs was found!")
            return

        args.out_dir = os.path.join(str(cig.graph_directory), args.out_dir)

    if cig is not None:
        args.graph_path = os.path.join(cig.graph_directory, cig.graph_file_name)
        print("[I] cig.graph_directory", cig.graph_file_name)
        print("[I] cig.graph_directory", cig.graph_directory)

    print("  [*] graph_path", args.graph_path)
    print("  [*] out_dir", args.out_dir)

    if args.graph_path is None:
        SystemController.end_doing_information_graph(system_settings)
        logging.error("There are no information graphs without sub graphs!")
        return

    with open(args.graph_path, "rb") as f:
        graph = pickle.load(f)

    ms_v = args.min_sim_value
    mt_v = args.min_tg_value
    res_dir = f"min-sim-{ms_v}__min-tg-{mt_v}"
    out_dir = os.path.join(args.out_dir, res_dir)

    extractor = BFSSubGraphsExtractor(
        g=graph, min_sim=args.min_sim_value, min_tan=args.min_tg_value
    )
    extractor.extract()
    extractor.save_sub_graphs(out_dir=out_dir)

    if cig is not None:
        extractor.store_sub_graphs_to_database(
            cont_info_graph=cig, out_dir=out_dir, type_str=args.type
        )

        if args.remove_previous:
            delete_subgraphs(instead_of=cig, type_str=args.type)

    SystemController.end_doing_information_sub_graph(system_settings)


if __name__ == "__main__":
    main()
