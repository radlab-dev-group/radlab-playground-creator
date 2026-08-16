import logging
import os
import queue
import threading
from typing import List, Optional, Sequence

from django import db
from sentence_transformers.cross_encoder import CrossEncoder

from creator.controllers.news import NewsController
from creator.models import GeneratedNews, NewsSubPage


def load_cross_encoder_model(
    model_name: str = "radlab/polish-cross-encoder",
    device: Optional[str] = None,
) -> CrossEncoder:
    """
    Load Cross-Encoder model with optional device selection.
    """
    if device is None:
        device = os.environ.get("CROSS_ENCODER_DEVICE", "auto")

    logging.info(f"Loading CE model {model_name} on device: {device}...")
    model = CrossEncoder(model_name, device=device)
    logging.info(f"Model {model_name} is loaded.")
    return model


def generate_news_parallel(
    news_controller: NewsController,
    articles: Sequence[NewsSubPage],
    cross_encoder_model: Optional[CrossEncoder] = None,
    num_workers: int = 1,
) -> List[GeneratedNews]:
    """
    Generate news for a list of subpages in parallel using worker threads.

    :param news_controller: NewsController instance to generate news
    :param articles: Collection of NewsSubPage instances to summarize
    :param cross_encoder_model: Optional CrossEncoder model for similarity calculation
    :param num_workers: Number of worker threads (default: 1)
    :return: List of successfully generated GeneratedNews objects
    """
    if not articles:
        return []

    all_generated_news: List[GeneratedNews] = []
    total_count = len(articles)
    num_workers = max(1, num_workers)
    tasks_queue: queue.Queue = queue.Queue()
    results_lock = threading.Lock()

    def worker():
        while True:
            item = tasks_queue.get()
            if item is None:
                tasks_queue.task_done()
                break

            idx, news_sub_page = item
            try:
                logging.info(
                    f"[{idx + 1}/{total_count}] "
                    f"Generating news for {news_sub_page.news_url}"
                )
                generated_news = news_controller.generate_news(
                    news_sub_page=news_sub_page,
                    cross_encoder_sim_model=cross_encoder_model,
                )
                if generated_news is None:
                    logging.warning(
                        f"Problem occurred while generating news for "
                        f"{news_sub_page.news_url}"
                    )
                else:
                    with results_lock:
                        all_generated_news.append(generated_news)
            except Exception as e:
                logging.error(
                    f"Error while generating news for {news_sub_page.news_url}: {e}"
                )
            finally:
                tasks_queue.task_done()
                db.close_old_connections()

    threads = []
    for _ in range(num_workers):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    for idx, news_sub_page in enumerate(articles):
        tasks_queue.put((idx, news_sub_page))

    for _ in range(num_workers):
        tasks_queue.put(None)

    for t in threads:
        t.join()

    return all_generated_news
