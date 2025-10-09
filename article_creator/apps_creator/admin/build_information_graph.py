import os
import tqdm
import json
import django
import argparse

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from system.controllers import SystemController
from creator.models import SingleDaySummary, SimilarClusters, Cluster
from creator.controllers.graph import InformationGraphController


def get_all_days_summaries() -> list[SingleDaySummary]:
    return list(
        SingleDaySummary.objects.filter(is_active=True).order_by("-when_generated")
    )


def get_clusters_for_day(day: SingleDaySummary) -> list[Cluster]:
    return list(Cluster.objects.filter(is_active=True, clustering=day.clustering))


def prepare_parser(desc=""):
    p = argparse.ArgumentParser(description=desc)

    p.add_argument(
        "-o", "--out-dir", dest="out_dir", default=None, required=False, type=str
    )
    p.add_argument("--limit", dest="limit", required=False, default=None, type=int)

    p.add_argument("--use-cache", dest="use_cache", action="store_true")
    p.add_argument("--add-d2d-edges", dest="add_d2d_edges", action="store_true")
    p.add_argument(
        "--personalize-days", dest="personalize_days", action="store_true"
    )

    p.add_argument("--standardize", dest="standardize", action="store_true")
    p.add_argument(
        "--normalize-tangens", dest="normalize_tangens", action="store_true"
    )
    p.add_argument(
        "--normalize-minmax", dest="normalize_minmax", action="store_true"
    )

    return p


def main(argv=None):
    args = prepare_parser().parse_args(argv)

    system_settings = SystemController.get_system_settings()
    if system_settings.doing_news_information_graph:
        return

    SystemController.begin_doing_information_graph(system_settings)

    icg_controller = InformationGraphController(
        out_dir=args.out_dir, use_cache=args.use_cache, try_to_load=True
    )
    if not icg_controller.is_loaded:
        sc_count = 0
        all_sc = SimilarClusters.objects.filter(is_active=True)
        with tqdm.tqdm(
            total=len(all_sc), desc="Preparing information graph"
        ) as pbar:
            for sc in all_sc:
                if args.limit is not None and 0 < args.limit <= sc_count:
                    break

                icg_controller.add_cluster_node(cluster=sc.source)
                icg_controller.add_cluster_node(cluster=sc.target)
                icg_controller.add_similarity_edge(similarity=sc)

                pbar.update(1)
                sc_count += 1

        print(json.dumps(icg_controller.summary(), indent=2, ensure_ascii=False))

        icg_controller.prepare_graph(
            normalize_tangens=args.normalize_tangens,
            normalize_minmax=args.normalize_minmax,
            standardize=args.standardize,
            tan_as_weights=True,
            minmax_as_weights=False,
        )

        icg_controller.store_continuous_information_graph_to_db(
            deactivate_other_info_graphs=True
        )

    # # Day-to-day edges
    if args.add_d2d_edges and not icg_controller.has_day_to_day_edges:
        # Fetch data
        d2c = {}
        days = get_all_days_summaries()
        with tqdm.tqdm(total=len(days), desc="Fetching data...") as pbar:
            for day in days:
                if day.day_to_summary not in d2c:
                    clusters = get_clusters_for_day(day=day)
                    d2c[day.day_to_summary] = clusters

        with tqdm.tqdm(total=len(days), desc="Adding daily connections") as pbar:
            prev_day = None
            prev_clusters = None
            for day in days:
                clusters = d2c[day.day_to_summary]
                if prev_day is not None:
                    icg_controller.add_day_to_day_edge(
                        clusters_from=clusters,
                        clusters_to=prev_clusters,
                    )
                prev_day = day
                prev_clusters = clusters
                pbar.update(1)
        icg_controller.has_day_to_day_edges = True

    # Daily personalization
    if args.personalize_days and not icg_controller.is_personalized:
        with tqdm.tqdm(total=len(days), desc="Personalizing daily graphs") as pbar:
            for day in days:
                icg_controller.personalize_day(
                    day=day, clusters=d2c[day.day_to_summary]
                )
                pbar.update(1)
        icg_controller.is_personalized = True

    SystemController.end_doing_information_graph(system_settings)


if __name__ == "__main__":
    main()
