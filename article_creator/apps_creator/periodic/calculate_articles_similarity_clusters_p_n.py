import os
import django
from tqdm import tqdm

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from system.controllers import SystemController
from creator.controllers.clustering import ClusteringSimilarityController


TEXT_COLUMN_NAME = "text"
METADATA_COLUMN_NAME = "metadata"


def main(argv=None):
    system_settings = SystemController.get_system_settings()
    if system_settings.doing_news_generation_for_yesterday_similarity:
        return

    SystemController.begin_public_yesterday_news_generation_similarity(
        system_settings
    )

    similarity_bias = 0.35
    n_most_sim_clusters = 10

    news_controller = ClusteringSimilarityController(
        clustering_config_path="configs/clusterer-config.json",
        device="cpu",
        store_to_db=True,
        n_most_sim_clusters=n_most_sim_clusters,
    )

    clusters = list(news_controller.get_clusters_without_similarity())
    with tqdm(total=len(clusters), desc="Calculating cluster similarity") as pbar:
        for cl_obj in clusters:
            similarities = news_controller.check_similarity_of_cluster(
                cluster=cl_obj,
                n_most_sim_clusters=n_most_sim_clusters,
                similarity_bias=similarity_bias,
                check_next=True,
                check_prev=True,
            )

            for direction_where, clusters_propositions in similarities.items():
                print("==" * 50)
                print("DIRECTION: ", direction_where)
                print("MOST SIM CLUSTERS")
                if clusters_propositions is None:
                    print(" -> no similarities for direction", direction_where)
                    continue

                for most_sim_clusters in clusters_propositions:
                    for msc in most_sim_clusters:
                        print(
                            f"  -> {msc.similarity_metric}("
                            f"{msc.source.pk}, {msc.target.pk}"
                            f")={msc.similarity_value}"
                        )
                        print("  -> msc.source.label_str=", msc.source.label_str)
                        print("  -> msc.target.label_str=", msc.target.label_str)
                        print("  -> msc.similarity_model=", msc.similarity_model)
                        print("  -> msc.is_active=", msc.is_active)
                        print("  -> msc.clustering=", msc.clustering)
                        print("  " + "- " * 20)
            pbar.update(1)

    SystemController.end_public_yesterday_news_generation_similarity(system_settings)


if __name__ == "__main__":
    main()
