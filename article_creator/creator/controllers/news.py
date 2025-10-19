import os
import json
import random
import hashlib
import logging
import datetime
import requests
import urllib.request

from bs4 import BeautifulSoup
from typing import List, Dict, Tuple, Any, Optional
from django.db.models import Q, QuerySet

from radlab_data.text.utils import TextUtils

from creator.models import SystemUser, GeneratedNews, Clustering, NewsMainPage

from general.api_utils import BasePublicApiInterface
from general.constants import DEFAULT_MODELS_CONFIG

from creator.models import (
    NewsCategory,
    NewsMainPage,
    NewsSubPage,
    GeneratedNews,
    FullGeneratedArticles,
    MainNewsPageContent,
    SingleDaySummary,
    Cluster,
    SampleClusterData,
    SimilarClusters,
)

from creator.serializers import (
    SingleDaySummarySerializer,
    ClusterSerializer,
    SimilarClustersSerializer,
)


class ProperNewsParameters:
    language: str = "pl"
    min_length: int = 125
    min_sim_to_original: float = 0.501
    max_num_of_generated_news: int = 4
    soft_num_of_generated_news: int = 2


class MainNewsController:
    def __init__(self, add_to_db: bool = True):
        self._add_to_db = add_to_db

    def get_add_category(
        self, name: str, display_name: str, description: str, order: int
    ) -> NewsCategory:
        if self._add_to_db:
            n_c, created = NewsCategory.objects.get_or_create(
                name=name, display_name=display_name, description=description
            )
            if created:
                n_c.order = order
                n_c.save()
            return n_c
        return NewsCategory(
            name=name, display_name=display_name, description=description
        )

    def get_add_main_public_url(
        self,
        main_url_str: str,
        begin_crawling_url: str,
        category: NewsCategory,
        index_page: bool,
        index_page_sse: bool,
        prepare_news: bool,
        show_news: bool,
        min_news_depth: int,
        include_paths_with: List[str],
        exclude_paths_with: List[str],
        news_url_ends_with: List[str],
        use_heuristic_to_extract_news: bool,
        use_heuristic_to_search_news_content: bool,
        use_heuristic_to_clear_news_content: bool,
        news_link_starts_with_main_url: bool,
        language: str,
        single_news_content_tag_name: str,
        single_news_content_tag_attr_name: str,
        single_news_content_tag_attr_value: str,
        remove_n_last_elems: int,
    ) -> NewsMainPage:
        if self._add_to_db:
            mp, created = NewsMainPage.objects.get_or_create(
                main_url=main_url_str,
                begin_crawling_url=begin_crawling_url,
                category=category,
                min_news_depth=min_news_depth,
                include_paths_with=include_paths_with,
                exclude_paths_with=exclude_paths_with,
                news_url_ends_with=news_url_ends_with,
                news_link_starts_with_main_url=news_link_starts_with_main_url,
                language=language,
                index_page=True,
            )
            if created:
                mp.index_page = index_page
                mp.index_page_sse = index_page_sse
                mp.prepare_news = prepare_news
                mp.show_news = show_news

            mp.remove_n_last_elems = remove_n_last_elems

            mp.use_heuristic_to_extract_news_content = use_heuristic_to_extract_news
            mp.use_heuristic_to_search_news_content = (
                use_heuristic_to_search_news_content
            )
            mp.use_heuristic_to_clear_news_content = (
                use_heuristic_to_clear_news_content
            )
            mp.single_news_content_tag_name = single_news_content_tag_name
            mp.single_news_content_tag_attr_name = single_news_content_tag_attr_name
            mp.single_news_content_tag_attr_value = (
                single_news_content_tag_attr_value
            )
            mp.save()

            return mp

        return NewsMainPage(
            show_news=show_news,
            remove_n_last_elems=remove_n_last_elems,
            main_url=main_url_str,
            begin_crawling_url=begin_crawling_url,
            category=category,
            prepare_news=True,
            include_paths_with=include_paths_with,
            exclude_paths_with=exclude_paths_with,
            news_url_ends_with=news_url_ends_with,
            use_heuristic_to_extract_news_content=use_heuristic_to_extract_news,
            news_link_starts_with_main_url=news_link_starts_with_main_url,
            single_news_content_tag_name=single_news_content_tag_name,
            single_news_content_tag_attr_name=single_news_content_tag_attr_name,
            single_news_content_tag_attr_value=single_news_content_tag_attr_value,
            language=language,
        )


class NewsController:
    MAX_PUBLIC_NEWS_CHAR_LENGTH = 10000
    MIN_ARTICLE_LEN_TO_GENERATE_NEWS = 150
    MAIN_NEWS_STREAM_PUBLIC_JSON_FIELD = "stream_news_free"
    MAIN_NEWS_STREAM_GENERATE_ARTICLE = "generate_article_from_text"
    MAIN_NEWS_CREATOR_GENERATE_ARTICLE = "generate_article_from_search_result"
    API_HEADER = {"Content-Type": "application/json; charset=utf-8"}

    MODEL_TO_NEWS_SIMILARITY_CALCULATION_CE = "radlab/polish-cross-encoder"

    def __init__(
        self,
        add_to_db: bool = True,
        seconds_prev_check: int = 0,
        models_config_path: str or None = DEFAULT_MODELS_CONFIG,
    ):
        """
        Initialize controller

        :param add_to_db: If set to True then results will be stored to database
        :param seconds_prev_check: Number of seconds to check content
        since the last check, default is 0 seconds (each call wil check content)
        """
        self._add_to_db = add_to_db
        self._seconds_prev_check = seconds_prev_check

        self._models_config = None
        self._models_config_path = models_config_path

        if models_config_path is not None and len(models_config_path.strip()):
            self._load_models_config(models_config_path)

    def download_news_main_pages_contents(
        self,
        user: SystemUser | None = None,
        debug_for_new_site: bool = False,
        add_html_to_db: bool = False,
    ) -> List[MainNewsPageContent]:
        """
        This is main function to download page content from each main url.
        The content will be downloaded when:
            - has not been downloaded yet
            - the time since last download is longer than time
        :return: Object of MainNewsPageContent
        """
        time_now = datetime.datetime.now()
        filter_opts = {"index_page": True}
        if user is not None:
            filter_opts["user"] = user

        if self._seconds_prev_check is not None and self._seconds_prev_check > 0:
            time_to_check = time_now - datetime.timedelta(
                seconds=self._seconds_prev_check
            )
            main_pages = NewsMainPage.objects.filter(
                Q(last_check__lte=time_to_check) | Q(last_check__isnull=True),
                **filter_opts,
            )
        else:
            main_pages = NewsMainPage.objects.filter(**filter_opts)
        main_news_page_contents = self._scrap_main_pages_content(
            list(main_pages),
            debug_for_new_site=debug_for_new_site,
            add_html_to_db=add_html_to_db,
        )
        return main_news_page_contents

    def download_news_subpages(
        self,
        main_pages_contents: List[MainNewsPageContent],
        add_html_to_db: bool = False,
    ) -> List[NewsSubPage]:
        news_subpages = []

        all_news_url_links = []
        for main_page_content in main_pages_contents:
            for news_url_link in main_page_content.extracted_urls:
                all_news_url_links.append([main_page_content, news_url_link])
        random.shuffle(all_news_url_links)
        for main_page_content, news_url_link in all_news_url_links:
            if main_page_content.main_page.news_link_starts_with_main_url:
                news_url_link_whole = news_url_link
            else:
                m_url = main_page_content.main_page.main_url.strip("/")
                n_url = news_url_link.strip("/")
                if (
                    n_url.startswith("http:")
                    or n_url.startswith("https:")
                    or n_url.startswith("www.")
                ):
                    news_url_link_whole = n_url
                else:
                    news_url_link_whole = m_url + "/" + n_url

            if self.__exists_news_sub_page(
                main_page=main_page_content.main_page,
                news_sub_page_url=news_url_link_whole,
            ):
                continue

            news_html = NewsContentGrabberController.download_page_content_html(
                url=news_url_link_whole
            )
            if news_html is None or not len(news_html):
                continue

            news_content = (
                NewsContentGrabberController.extract_content_from_news_page(
                    html_content=news_html, main_page=main_page_content.main_page
                )
            )
            if news_content is None:
                continue

            news_html = news_html if add_html_to_db else ""
            news_sub_page = self._add_news_sub_page(
                news_url=news_url_link_whole,
                main_page=main_page_content.main_page,
                page_content_html=news_html,
                page_content_txt=news_content,
            )
            if news_sub_page is not None:
                news_subpages.append(news_sub_page)
        return news_subpages

    @staticmethod
    def public_subpages_without_summarization() -> QuerySet[NewsSubPage]:
        """
        Returns list of subpages where:
          - main_page has prepare_news a set as True
          - subpage does not have the summary yet
        :return: List of NewsSubpage objects ordered by id
        """
        return NewsSubPage.objects.filter(
            main_page__prepare_news=True,
            has_generated_news=False,
            skip_subpage=False,
        ).order_by("id")

    @staticmethod
    def public_get_generated_news_for_date_range(
        begin_date: datetime.date, end_date: datetime.date
    ) -> QuerySet[GeneratedNews]:
        return GeneratedNews.objects.filter(
            show_news=True,
            when_generated__lte=end_date,
            when_generated__gte=begin_date,
        )

    def generate_news(
        self, news_sub_page: NewsSubPage, cross_encoder_sim_model=None
    ) -> GeneratedNews | None:
        assert self._models_config is not None
        if news_sub_page is None:
            return None

        if (
            len(news_sub_page.page_content_txt.strip())
            < self.MIN_ARTICLE_LEN_TO_GENERATE_NEWS
        ):
            if self._add_to_db:
                NewsSubPage.objects.filter(pk=news_sub_page.pk).update(
                    skip_subpage=True
                )
            return None

        orig_article_str = news_sub_page.page_content_txt.strip()
        gen_article_str, gen_time, model_name = self._generate_article_from_article(
            article_str=orig_article_str
        )

        news_sub_page.num_of_generated_news += 1
        if gen_article_str is None or not len(gen_article_str):
            if self._add_to_db:
                NewsSubPage.objects.filter(pk=news_sub_page.pk).update(
                    num_of_generated_news=news_sub_page.num_of_generated_news
                )
            return None

        article_language = self.__check_language(text=gen_article_str)

        gen_news_sim = None
        if cross_encoder_sim_model is not None:
            gen_news_sim = self._calculate_similarity_ce(
                gen_article_str=gen_article_str,
                orig_article_str=orig_article_str,
                cross_encoder_sim_model=cross_encoder_sim_model,
                ce_model_max_seq_len=514,
            )

        show_news = news_sub_page.main_page.show_news
        if self._add_to_db:
            gen_news = GeneratedNews.objects.create(
                generated_text=gen_article_str,
                generation_time=gen_time,
                language=article_language,
                model_used_to_generate_news=model_name,
                show_news=show_news,
                news_sub_page=news_sub_page,
                similarity_to_original=gen_news_sim,
            )
        else:
            gen_news = GeneratedNews(
                generated_text=gen_article_str,
                generation_time=gen_time,
                language=article_language,
                model_used_to_generate_news=model_name,
                show_news=show_news,
                news_sub_page=news_sub_page,
                similarity_to_original=gen_news_sim,
            )

        self.__validate_news_and_subpage(
            gen_news, news_sub_page, soft_validation=True
        )

        NewsSubPage.objects.filter(pk=news_sub_page.pk).update(
            num_of_generated_news=news_sub_page.num_of_generated_news
        )

        return gen_news

    def generate_article_full_from_news_list(
        self,
        user_query: str,
        news_list: List[GeneratedNews],
        new_article_type: str | None,
        query_response_id: int | None,
        model_name: str,
    ) -> FullGeneratedArticles | None:
        news_texts = [n.generated_text for n in news_list]
        ep_data = {
            "user_query": user_query,
            "texts": news_texts,
            "article_type": new_article_type,
            "model_name": model_name,
            "top_k": 50,
            "top_p": 0.99,
            "temperature": 0.65,
            "typical_p": 1.0,
            "repetition_penalty": 1.07,
            "max_new_tokens": 3560,
        }

        model_host = self._models_config[self.MAIN_NEWS_CREATOR_GENERATE_ARTICLE][
            "model_hosts"
        ][0]
        prepare_article_ep = self._models_config[
            self.MAIN_NEWS_CREATOR_GENERATE_ARTICLE
        ]["ep"]["create_article_from_news_list"]
        ep_url = f"{model_host.strip('/')}/{prepare_article_ep.strip('/')}"
        ep_response = BasePublicApiInterface.general_call_post(
            host_url=None,
            endpoint=ep_url,
            data=None,
            json_data=ep_data,
            headers=self.API_HEADER,
            login_url=None,
        )

        if "response" not in ep_response:
            return None

        ep_gen_time = datetime.timedelta(seconds=ep_response["generation_time"])

        article_str = ""
        if type(ep_response["response"]) == dict:
            article_str = ep_response["response"]["article_text"]
        elif type(ep_response["response"]) == str:
            article_str = ep_response["response"]
        else:
            raise Exception("Unknown response type!")

        news_list_ids = [n.pk for n in news_list]
        full_g_art = FullGeneratedArticles.objects.create(
            article_str=article_str,
            generation_time=ep_gen_time,
            model_used_to_generate=model_name,
            user_query=user_query,
            based_on_news=news_list_ids,
            sse_query_response_id=query_response_id,
            article_type=new_article_type,
        )

        return full_g_art

    def __validate_news_and_subpage(
        self, news: GeneratedNews, subpage: NewsSubPage, soft_validation: bool
    ):
        """
        Validates generated news content and updates both
        news and subpage status based on quality criteria.

        This private method performs comprehensive validation of generated
        news articles against predefined quality parameters and manages
        the visibility and database state of both the news item
        and its associated subpage.

        Args:
            news (GeneratedNews): The generated news object to validate.
            Must contain attributes: show_news, similarity_to_original,
            generated_text, and language.
            subpage (NewsSubPage): The news subpage object associated
            with the news item. Must contain attributes:
            num_of_generated_news and skip_subpage.
            soft_validation (bool): If True, applies more lenient validation
            rules using soft_num_of_generated_news threshold. If False,
            uses stricter validation with a max_num_of_generated_news threshold.

        Validation Criteria:
            - Similarity threshold: news.similarity_to_original must meet
            ProperNewsParameters.min_sim_to_original
            - Length requirement: news.generated_text length must meet
            ProperNewsParameters.min_length
            - Language matching: news.language must match
            ProperNewsParameters.language (if specified)

        Behavior:
            - If news initially has show_news=False, validation is skipped entirely
            - Failed validation sets news.show_news=False
            and subpage.has_generated_news=False
            - In strict mode: subpage.skip_subpage=True
            if num_of_generated_news >= max_num_of_generated_news
            - In soft mode: overrides validation failure
            if num_of_generated_news >= soft_num_of_generated_news
            - Database updates are performed only if self._add_to_db is True

        Side Effects:
            - Updates NewsSubPage.has_generated_news in database
            - Updates GeneratedNews.show_news in a database (when validation
            fails and self._add_to_db is True)
            - May set subpage.skip_subpage=True in strict validation mode

        Returns:
            None: This method performs validation and updates objects in-place.
        """

        if not news.show_news:
            return

        show_news = True
        if news.similarity_to_original < ProperNewsParameters.min_sim_to_original:
            show_news = False
        elif len(news.generated_text) < ProperNewsParameters.min_length:
            show_news = False
        elif (
            news.language is not None
            and len(news.language)
            and news.language != ProperNewsParameters.language
        ):
            show_news = False

        has_generated_news = True
        if not show_news:
            news.show_news = False
            has_generated_news = False
            if not soft_validation:
                if (
                    subpage.num_of_generated_news
                    >= ProperNewsParameters.max_num_of_generated_news
                ):
                    subpage.skip_subpage = True
            else:
                if (
                    subpage.num_of_generated_news
                    >= ProperNewsParameters.soft_num_of_generated_news
                ):
                    news.show_news = True
                    has_generated_news = True

        NewsSubPage.objects.filter(pk=subpage.pk).update(
            has_generated_news=has_generated_news
        )

        if not show_news and self._add_to_db:
            GeneratedNews.objects.filter(pk=news.pk).update(show_news=False)

    @staticmethod
    def __check_language(text: str) -> str:
        return TextUtils.text_language(text)

    @staticmethod
    def _calculate_similarity_ce(
        gen_article_str,
        orig_article_str,
        cross_encoder_sim_model,
        ce_model_max_seq_len: int,
    ) -> float:

        s1 = gen_article_str[: ce_model_max_seq_len - 1]
        s2 = orig_article_str[: ce_model_max_seq_len - 1]
        sim_val = cross_encoder_sim_model.predict([(s1, s2)])
        sim_val = sim_val[0] if len(sim_val) else 0.0
        return sim_val

    def _generate_article_from_article(
        self, article_str: str
    ) -> (str | None, float | None, str | None):
        if len(article_str) > self.MAX_PUBLIC_NEWS_CHAR_LENGTH:
            article_str = article_str[: self.MAX_PUBLIC_NEWS_CHAR_LENGTH]

        model_name = self._models_config[self.MAIN_NEWS_STREAM_GENERATE_ARTICLE][
            "model_name"
        ]
        model_host = self._models_config[self.MAIN_NEWS_STREAM_GENERATE_ARTICLE][
            "model_hosts"
        ][0]
        prepare_article_ep = self._models_config[
            self.MAIN_NEWS_STREAM_GENERATE_ARTICLE
        ]["ep"]["prepare_article"]
        ep_url = f"{model_host.strip('/')}/{prepare_article_ep.strip('/')}"

        ep_data = {
            "text": article_str,
            "model_name": model_name,
            "top_k": 0,
            "top_p": 0.90,
            "max_new_tokens": 256,
            "temperature": 0.6,
            "typical_p": 1.0,
            "repetition_penalty": 1.00,
        }

        ep_response = BasePublicApiInterface.general_call_post(
            host_url=None,
            endpoint=ep_url,
            data=None,
            json_data=ep_data,
            headers=self.API_HEADER,
            login_url=None,
        )
        if "response" not in ep_response:
            return None, None, None

        article_text = ep_response["response"].get("article_text", None)
        gen_time = ep_response.get("generation_time", None)

        if gen_time is not None:
            gen_time = datetime.timedelta(seconds=gen_time)

        return article_text, gen_time, model_name

    def _load_models_config(self, models_config_path):
        self._models_config = json.load(open(models_config_path, "rt")).get(
            self.MAIN_NEWS_STREAM_PUBLIC_JSON_FIELD
        )
        assert (
            self._models_config is not None
        ), f"Cannot find {self.MAIN_NEWS_STREAM_PUBLIC_JSON_FIELD} in config!"

    def _add_news_sub_page(
        self,
        news_url: str,
        main_page: NewsMainPage,
        page_content_html: str,
        page_content_txt: str,
    ) -> NewsSubPage | None:
        if self._add_to_db:
            return NewsSubPage.objects.create(
                news_url=news_url,
                page_content_html=page_content_html,
                page_content_txt=page_content_txt,
                main_page=main_page,
            )
        return NewsSubPage(
            news_url=news_url,
            page_content_html=page_content_html,
            page_content_txt=page_content_txt,
            main_page=main_page,
        )

    def _scrap_main_pages_content(
        self,
        main_pages: List[NewsMainPage],
        debug_for_new_site: bool = False,
        add_html_to_db: bool = False,
    ) -> List[MainNewsPageContent]:
        main_news_page_contents = []
        for main_page in main_pages:
            main_page_content = self._scrap_main_page_content(
                main_page,
                debug_for_new_site=debug_for_new_site,
                add_html_to_db=add_html_to_db,
            )
            if main_page_content is not None:
                main_news_page_contents.append(main_page_content)
        return main_news_page_contents

    def _scrap_main_page_content(
        self,
        main_page: NewsMainPage,
        debug_for_new_site: bool = False,
        add_html_to_db: bool = False,
    ) -> MainNewsPageContent | None:
        logging.info(f"Scraping www page: {main_page.begin_crawling_url}")
        content_html = NewsContentGrabberController.download_page_content_html(
            url=main_page.begin_crawling_url
        )
        if content_html is None or not len(content_html):
            return None

        extracted_urls = NewsContentGrabberController.extract_news_urls_from_html(
            html=content_html, main_page=main_page
        )
        if debug_for_new_site:
            print(json.dumps(extracted_urls, indent=2, ensure_ascii=False))
            return None

        content_hash = self.__prepare_hash_main_page_content(extracted_urls)
        if content_hash is None or not len(content_html):
            return None

        if self.__exists_news_page_content(content_hash):
            return None

        content_html = content_html if add_html_to_db else ""
        page_content = self._add_main_page_content(
            main_page, content_hash, content_html, extracted_urls
        )
        logging.info(f"Content hash: {content_hash}")

        return page_content

    def _add_main_page_content(
        self, main_page, content_hash, content_html, extracted_urls
    ) -> MainNewsPageContent:
        if self._add_to_db:
            return MainNewsPageContent.objects.create(
                main_page=main_page,
                content_hash=content_hash,
                html_content=content_html,
                extracted_urls=extracted_urls,
            )
        return MainNewsPageContent(
            main_page=main_page,
            content_hash=content_hash,
            html_content=content_html,
            extracted_urls=extracted_urls,
        )

    @staticmethod
    def __prepare_hash_main_page_content(extracted_urls: List[str]) -> str | None:
        urls_as_str = "".join(extracted_urls).strip()
        if not len(urls_as_str):
            return None
        urls_as_str = urls_as_str.encode()
        return hashlib.md5(urls_as_str).hexdigest()

    @staticmethod
    def __exists_news_page_content(content_hash: str) -> bool:
        return MainNewsPageContent.objects.filter(content_hash=content_hash).exists()

    @staticmethod
    def __exists_news_sub_page(
        main_page: NewsMainPage, news_sub_page_url: str
    ) -> bool:
        return NewsSubPage.objects.filter(
            main_page=main_page, news_url=news_sub_page_url
        ).exists()


class NewsContentGrabberController:
    MOZILLA_HEADER_SIMPLE = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; "
        "Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0"
    }

    MOZILLA_HEADER_FULL = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
    }

    def __init__(self):
        pass

    @staticmethod
    def extract_news_content_from_selected_tag(
        soup,
        main_page: NewsMainPage,
        html_tag_name: str,
        html_tag_attr_name: str | None,
        html_tag_attr_val: str | None,
    ) -> str | None:
        if html_tag_attr_name is None:
            elems = soup.find_all(html_tag_name)
        else:
            elems = soup.find_all(
                html_tag_name, attrs={html_tag_attr_name: html_tag_attr_val}
            )

        whole_text = ""
        for e in elems:
            elem_text = e.text
            if main_page.use_heuristic_to_clear_news_content:
                elem_text_new = ""
                elem_text_spl = elem_text.split("\n")
                for e_t_spl in elem_text_spl:
                    skip_elem = (
                        NewsContentGrabberController.__skip_partial_news_content(
                            paragraph_text=e_t_spl
                        )
                    )
                    if skip_elem or not len(e_t_spl.strip()):
                        continue
                    elem_text_new += " " + e_t_spl + "\n"
                elem_text = elem_text_new.strip()
            whole_text += elem_text + "\n"

        if not len(whole_text.strip()):
            return None

        return whole_text

    @staticmethod
    def extract_news_content_from_paragraphs(soup, main_page: NewsMainPage) -> str:
        whole_content = ""
        if main_page.use_heuristic_to_search_news_content:
            all_paragraphs = soup.find_all("p", class_=True)
            all_paragraphs.extend(soup.find_all("p"))
        else:
            all_paragraphs = soup.find_all("p")

        for par in all_paragraphs:
            if par.text is None or not len(par.text.strip()):
                continue
            par_text = par.text.strip()
            if not len(par_text):
                continue
            skip_paragraph = False
            # use_heuristic_to_search_news_content
            if main_page.use_heuristic_to_search_news_content:
                if par_text in whole_content:
                    skip_paragraph = True
                else:
                    skip_paragraph = (
                        NewsContentGrabberController.__h_search_news_content(
                            paragraph_tag_elem=par
                        )
                    )
            # use_heuristic_to_clear_news_content
            if not skip_paragraph and main_page.use_heuristic_to_clear_news_content:
                skip_paragraph = (
                    NewsContentGrabberController.__skip_partial_news_content(
                        paragraph_text=par_text
                    )
                )
            if skip_paragraph:
                continue
            if len(par_text.strip()):
                whole_content += par_text + "\n"
            for li in par.find_all("li"):
                whole_content += "\t" + li.text + "\n"
            whole_content += "\n"

        return whole_content

    @staticmethod
    def extract_content_from_news_page(
        html_content: str, main_page: NewsMainPage
    ) -> str | None:
        if html_content is None or not len(html_content.strip()):
            return None

        soup = BeautifulSoup(html_content, features="lxml")
        if main_page.single_news_content_tag_name is not None and len(
            main_page.single_news_content_tag_name
        ):
            whole_content = (
                NewsContentGrabberController.extract_news_content_from_selected_tag(
                    soup=soup,
                    main_page=main_page,
                    html_tag_name=main_page.single_news_content_tag_name,
                    html_tag_attr_name=main_page.single_news_content_tag_attr_name,
                    html_tag_attr_val=main_page.single_news_content_tag_attr_value,
                )
            )
        else:
            whole_content = (
                NewsContentGrabberController.extract_news_content_from_paragraphs(
                    soup=soup, main_page=main_page
                )
            )

        if whole_content is None:
            return None

        whole_content = whole_content.strip()
        if main_page.use_heuristic_to_extract_news_content and len(whole_content):
            whole_content = (
                NewsContentGrabberController.prepare_news_content_with_heuristics(
                    content=whole_content
                )
            )

        whole_content = whole_content.strip()
        if (
            main_page.remove_n_last_elems is not None
            and main_page.remove_n_last_elems > 0
        ):
            spl_content = whole_content.split("\n")
            if len(spl_content) < main_page.remove_n_last_elems:
                return ""
            whole_content = "\n".join(spl_content[: -main_page.remove_n_last_elems])
            whole_content = whole_content.strip()

        return whole_content

    @staticmethod
    def __h_search_news_content(paragraph_tag_elem) -> bool:
        class_names = [
            "comment",
            "author",
            "twitter",
            "related-with-lead-article__lead",
        ]
        if paragraph_tag_elem.parent is not None:
            parent = paragraph_tag_elem.parent
            if parent.has_attr("class"):
                for p_c_name in parent["class"]:
                    for c in class_names:
                        if c.lower() in p_c_name.lower():
                            return True
        return False

    @staticmethod
    def __skip_partial_news_content(paragraph_text: str) -> bool:
        phrases_to_clear = [
            "\n\n\n",
            "eby widzieć komentarze musisz",
            "zobacz także: ",
            "zobacz także: ",
            "Czytaj więcej: ",
            "Czytaj więcej: ",
            "Przeczytaj także: ",
            "Przeczytaj także: ",
            "Czytaj także: ",
            "Czytaj także: ",
            "Czytaj też: ",
            "Czytaj też: ",
            "Czytaj również: ",
            "Czytaj również: ",
            "alszy ciąg tekstu pod materiałem wideo",
            "z Google Play lub App Store",
            "w LEX:",
            "antyweb",
            "Czytaj także na Prawo.pl:",
            "Sprawdź również książkę:",
            "Dalsza część artykułu pod materiałem wideo",
            "Apteczka prawna - Lex bez łez",
            "Korzystanie z portalu oznacza akceptację Regulaminu.",
            "Copyright by INTERIA.PL 1999-2024.",
            "Hej, jesteśmy na Google News - Obserwuj to, co ważne w techu",
            "Zmień miejscowość",
            "Popularne miejscowości",
            "Interia Wydarzenia na Facebooku",
            "INTERIA.PL/PAP",
            "Otwórz galerię Na Gazeta.pl",
            "Polsat News",
            "/ Agencja Wyborcza.pl",
            "Fot. REUTERS",
            "Fot. Policja",
            "Fot. DALL·E 3",
            "Fot. Warszawa w Pigułce",
            "Fot. Pixabay",
            "Capital Media S.C. ul. Grzybowska 87, 00-844 Warszawa",
            "Kontakt z redakcją: [" "Zasady są proste - my podajemy nazwisko",
            "TVN24 | ",
            "TVN Meteo |",
            "Stock image from ",
            "Dziennikarz Business Insider Polska",
            "Ładuję…",
            "rozwiązać ten quiz",
            "Trwa ładowanie wpisu:",
            "Listen on Spreaker.",
            "ZOBACZ:",
            "WIDEO:",
            "niezależna.pl",
            "ZAMÓW »",
            " »",
            "CZYTAJ TERAZ",
            "Czytaj więcej",
            "Sprawdź gdzie kupisz Gazetę",
            "Razem ratujmy niezależne media!",
            "Gazeta Polska: ",
            "Zamów już TERAZ!",
            "©",
            "Teraz możesz docenić pracę dziennikarzy i dziennikarek."
            "Polsatnews.pl",
            "Daj napiwek autorowi",
            "Nie przegap ważnej informacji",
            "Skorzystaj z naszego bota >>",
            "Sprawdź: ",
            "Prowadzisz firmę?",
            "Dołącz do naszego katalogu!",
            "Skopiuj link",
            "Podziel się na Facebooku",
            "Podziel się na X",
            "WiadomościŚwiat",
            "Redakcja Wiadomości",
            "Napisz do autora",
            "PAP/",
            "Zobacz galerię",
            "Dalszy ciąg artykułu pod wideo",
            "Najważniejsze wiadomości z kraju i ze świata",
            "Jesteś świadkiem ciekawego wydarzenia? Skontaktuj się z nami",
            "Join visionaries from Precursor Ventures",
            "Реклама:",
            "РЕКЛАМА:",
            "Фото-",
            "Фото -",
            "Фото:",
            "Фото :",
        ]
        par_text_lower = paragraph_text.lower()
        for p2c in phrases_to_clear:
            if p2c.lower() in par_text_lower:
                return True
        return False

    @staticmethod
    def prepare_news_content_with_heuristics(content: str):
        clear_content = ""
        for c in content.split("\n"):
            clear_content += c.strip() + "\n"
        content = clear_content.strip()

        phrases = [
            "------",
            "Źródło: Fakt",
            "Copyright © (treści)",
            "Dodaj do ulubionych",
            "Źródło: WP",
            "Źródło: RMF FM",
            "źródło: BBC",
            "Źródło: PAP",
            "Źródło: PAP",
            "Źródło: IMGW",
            "Źródło: IMGW",
            "Źródło: TVN24",
            "Źródło: TVN24",
            "Źródło: tvnmeteo.pl",
            "Źródło zdjęcia głównego:",
            "Źródło: Polsat News",
            "Źródło: WP Wiadomości",
            "Źródło: New York Post",
            "Źródło: AP News",
            "Źródło: BBC",
            "Źródło: Deutsche",
            "Źródło: Reuters",
            "Źródło: Twitter",
            "Źródło: The",
            "Źródło: Radio Zet",
            "Autorka/Autor:",
            "Kobieta\nKobieta",
            "Kobieta\nWG",
            "Kobieta\nKO",
            "KO\nKobieta",
            "WG\nKobieta",
            "WG\nSport",
            "Sport\nSport",
            "WG\nWG",
            "WG\npolsat sport",
            "Sport\nBiznes",
            "Biznes\nBiznes",
            "Sport\nSport",
            "Biznes\nSport",
            "Biznes\nKobieta",
            "Kobieta\nBiznes",
            "polsat sport\npolsat sport",
            "Kobieta\nSport",
            "Sport\nKobieta",
            "Kobieta\nSport",
            "Sport\nWG",
            "Sport\npolsat sport",
            "Kobieta\npolsat sport",
            "WP Wiadomości  na:",
            "Źródło: IMGW",
            "\n\n\n\n\n",
            "(PAP)",
            "(PAP Biznes)",
            # "PAP",
            "Dziennikarka prasowa i internetowa. Z w",
            "osk/",
            "mal/",
            "twi/",
            "mce/",
            "mhr/",
            "gaw/",
            "osz/",
            "mms/",
            "ndz/",
            "nl/",
            "mja/",
            "sdd/",
            "adj/",
            "fit/",
            "ap/",
            "pr/",
            "MKZ",
            "Karol Darmoros/Vatican News, Republika",
            "Korzystanie z portalu oznacza akceptację regulaminu",
            "Źródło: Republika",
            "Źródło: Radio Republika",
            "Źródło: AM ART-MEDIA",
            "Telewizja Republika S.A.",
            "Dziękujemy, że przeczytałaś/eś nasz artykuł do końca.",
            "źr. ",
            "Źródło:",
            "Mat. oryginalny:",
        ]
        for p in phrases:
            content = content.split(p)[0]

        spl_content = content.strip().split("\n")
        if len(spl_content) > 6:
            deep_phrases = [
                "Czytaj też:",
                "Czytaj także:",
                "Czytaj również:",
                "Polecany artykuł:",
                "Przeczytaj też:",
                "Przeczytaj także:",
                "Przeczytaj również:",
                "Nie przegap:",
                "dzięki wyświetlaniu reklam",
            ]
            spl_begin = spl_content[:5]
            spl_end = spl_content[5:]
            for s_e in spl_end:
                stop_merging = False
                s_e_lower = s_e.lower()
                for s_e_dp in deep_phrases:
                    if s_e_dp.lower() in s_e_lower.lower() in s_e_lower:
                        stop_merging = True
                        break
                if stop_merging:
                    break
                spl_begin.append(s_e)
            content = "\n".join(spl_begin)
        return content

    @staticmethod
    def download_page_content_html(url: str) -> str | None:
        logging.info(f"Downloading news content: {url}")

        req = urllib.request.Request(
            url=url, headers=NewsContentGrabberController.MOZILLA_HEADER_FULL
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                resp_read = response.read()
                return resp_read
        except Exception as e:
            try:
                logging.warning("Catched exception while downloading news content")
                logging.warning(e)
                logging.warning("Trying to download content with request.get")
                resp_read = requests.get(url, timeout=10)
                logging.info(resp_read.text)
            except Exception as e:
                logging.error(f"Error during downloading {url}: {e}")

            return None

    @staticmethod
    def extract_news_urls_from_html(html: str, main_page: NewsMainPage) -> List[str]:
        if not len(html.strip()):
            return []

        urls = []
        soup = BeautifulSoup(html, features="lxml")
        for link in soup.find_all("a", href=True):
            link_href_str = link.get("href", "").strip()
            if NewsContentGrabberController.skip_url(
                link_href=link_href_str,
                main_page=main_page,
            ):
                continue
            if link_href_str not in urls:
                urls.append(link_href_str)
        return urls

    @staticmethod
    def skip_url(link_href: str, main_page: NewsMainPage):
        if link_href is None or not len(link_href):
            return True

        if main_page.news_link_starts_with_main_url:
            if not link_href.startswith(main_page.main_url):
                return True

        if main_page.min_news_depth > 0:
            sub_dirs = [
                x
                for x in os.path.normpath(link_href).split(os.path.sep)
                if len(x.strip())
            ]
            if len(sub_dirs) < main_page.min_news_depth:
                return True

        for ph_skip in main_page.exclude_paths_with:
            if ph_skip in link_href:
                return True

        if len(main_page.news_url_ends_with):
            for e_with in main_page.news_url_ends_with:
                e_with = e_with.strip().lower()
                if len(e_with) and link_href.lower().endswith(e_with):
                    return False
            return True

        if len(main_page.include_paths_with):
            for ph_accept in main_page.include_paths_with:
                if len(ph_accept) and ph_accept in link_href:
                    return False
            return True

        return False


class NewsControllerSimple:
    def __init__(self):
        pass

    @staticmethod
    def get_main_page(main_url: str) -> QuerySet[NewsMainPage]:
        return NewsMainPage.objects.filter(main_url__startswith=main_url)

    @staticmethod
    def get_main_page_contents(
        main_page: NewsMainPage,
    ) -> QuerySet[MainNewsPageContent]:
        return MainNewsPageContent.objects.filter(main_page=main_page)

    @staticmethod
    def get_main_page_contents_pk_list(
        main_page: NewsMainPage,
    ) -> list[Any]:
        return list(
            MainNewsPageContent.objects.filter(main_page=main_page)
            .order_by("-pk")
            .values_list("pk", flat=True)
        )

    @staticmethod
    def get_main_page_by_id(main_page_id) -> Optional[MainNewsPageContent]:
        """
        Retrieve a ``MainNewsPageContent`` instance by its primary key.

        Returns the object if it exists, otherwise ``None``.

        Args:
            main_page_id: Primary key of the desired ``MainNewsPageContent``.

        Returns:
            The matching ``MainNewsPageContent`` object or ``None`` if not found.
        """
        return MainNewsPageContent.objects.filter(pk=main_page_id).first()

    @staticmethod
    def get_main_page_subpages(main_page: NewsMainPage) -> QuerySet[NewsSubPage]:
        return NewsSubPage.objects.filter(main_page=main_page)

    @staticmethod
    def get_main_page_subpages_pk_list(main_page: NewsMainPage) -> list[Any]:
        return list(
            NewsSubPage.objects.filter(main_page=main_page)
            .order_by("-pk")
            .values_list("pk", flat=True)
        )

    @staticmethod
    def get_main_page_subpage_by_id(main_page_id) -> Optional[NewsSubPage]:
        return NewsSubPage.objects.filter(pk=main_page_id).first()

    @staticmethod
    def get_public_news_categories_with_pages() -> (
        List[Tuple[NewsCategory, List[NewsMainPage]]]
    ):
        out_res = []
        publ_cat = NewsControllerSimple.get_public_news_categories()
        for c in publ_cat:
            out_res.append(
                (
                    c,
                    NewsMainPage.objects.filter(
                        category=c, prepare_news=True, show_news=True
                    ),
                )
            )
        return out_res

    @staticmethod
    def get_public_news_categories() -> QuerySet[NewsCategory]:
        return NewsCategory.objects.all().order_by("order")

    @staticmethod
    def get_public_last_days_articles_from_sites(
        sites: List[str], last_days: int = 1
    ) -> Tuple[QuerySet[GeneratedNews], List[str], Dict, Dict]:
        begin_datetime = datetime.datetime.now() - datetime.timedelta(days=last_days)
        actual_last_subpages = NewsSubPage.objects.filter(
            when_crawled__gte=begin_datetime
        ).values_list("pk", "news_url")

        last_subpages_ids = [a[0] for a in actual_last_subpages]
        main_urls = [a[1] for a in actual_last_subpages]
        last_generated_news = GeneratedNews.objects.filter(
            show_news=True,
            news_sub_page__in=last_subpages_ids,
            news_sub_page__main_page__main_url__in=sites,
        )
        news_id_with_category = last_generated_news.values_list(
            "pk",
            "news_sub_page__main_page__category__name",
            "news_sub_page__news_url",
        )
        news_to_category = {n[0]: n[1] for n in news_id_with_category}
        news_to_url = {n[0]: n[2] for n in news_id_with_category}

        return (
            last_generated_news,
            list(set(main_urls)),
            news_to_category,
            news_to_url,
        )

    def news_from_all_categories(
        self,
        news_in_category: int,
        filter_pages: Dict[str, List] | None = None,
        polarity_3c: str | None = None,
        pli_from: int | None = None,
        pli_to: int | None = None,
    ) -> Dict[str, List[GeneratedNews]]:
        if filter_pages is None:
            filter_pages = {}

        filter_options = {}
        if polarity_3c is not None:
            filter_options["polarity_3c"] = polarity_3c
        if pli_from is not None:
            filter_options["pli_value__gte"] = pli_from
        if pli_to is not None and pli_from is not None:
            filter_options["pli_value__lte"] = pli_to

        return self.return_news_with_filter_options(
            news_in_category=news_in_category,
            filter_options=filter_options,
            filter_pages=filter_pages,
            similarity_opts=None,
        )

    def news_from_all_categories_to_check(
        self,
        news_in_category: int,
        filter_pages: Dict[str, List] | None,
        min_sim_to_orig: float,
        max_sim_to_orig: float,
        min_article_text_length: int,
    ):
        if filter_pages is None:
            filter_pages = {}

        # A lot of news to check
        filter_options = {"show_admin_message": True}
        lot_of_news = self.return_news_with_filter_options(
            news_in_category=news_in_category,
            filter_options=filter_options,
            filter_pages=filter_pages,
            similarity_opts={"min": min_sim_to_orig, "max": max_sim_to_orig},
        )

        merged = {}
        for c, news_list in lot_of_news.items():
            merged[c] = []
            for n in news_list:
                if n.similarity_to_original < min_sim_to_orig:
                    # Not similar
                    merged[c].append(n)
                elif n.similarity_to_original > max_sim_to_orig:
                    # Possibility of the plagiarism
                    merged[c].append(n)
                elif (
                    n.generated_text is None
                    or len(n.generated_text) < min_article_text_length
                ):
                    # The text is too short
                    merged[c].append(n)
                elif n.news_sub_page.num_of_generated_news > 1:
                    # News was generated multiple times
                    merged[c].append(n)
                elif n.generated_text.strip()[-1] not in [".", "!", "?", ";"]:
                    merged[c].append(n)
        return merged

    @staticmethod
    def return_news_with_filter_options(
        news_in_category: int,
        filter_options: dict,
        filter_pages: Dict[str, List] | None,
        similarity_opts: Dict[str, float] | None = None,
    ):
        generated_news = {}
        for category in NewsControllerSimple.get_public_news_categories():
            accept_pages = []
            filter_category_pages = filter_pages.get(category.name, None)
            if filter_category_pages and len(filter_category_pages):
                for page in filter_category_pages:
                    for p_url, p_status in page.items():
                        if p_status:
                            accept_pages.append(p_url)
            if not len(accept_pages):
                continue

            all_gen_news = GeneratedNews.objects.filter(
                news_sub_page__main_page__show_news=True,
                show_news=True,
                news_sub_page__main_page__main_url__in=accept_pages,
                news_sub_page__main_page__category=category,
                **filter_options,
            ).order_by("-news_sub_page__when_crawled")
            if similarity_opts is not None:
                # Note: hardcoded date to cut older news to check correctness
                all_gen_news = all_gen_news.filter(
                    news_sub_page__when_crawled__gte="2025-07-01"
                )

                min_sim = similarity_opts.get("min", 0.0)
                max_sim = similarity_opts.get("max", 1.0)
                all_gen_news = all_gen_news.filter(
                    Q(similarity_to_original__gte=max_sim)
                    | Q(similarity_to_original__lte=min_sim)
                )

            all_gen_news = all_gen_news[:news_in_category]

            generated_news[category.name] = all_gen_news
        return generated_news

    @staticmethod
    def get_news_by_ids(news_ids: list):
        return GeneratedNews.objects.filter(pk__in=news_ids)

    def do_action_on_news(self, news_id, action: str):
        try:
            news = GeneratedNews.objects.get(pk=news_id)
        except GeneratedNews.DoesNotExist:
            return

        if action == "regenerate":
            self.__regenerate_news(news)
        elif action == "hide":
            self.__hide_news(news)
        elif action == "hide_admin_msg":
            self.__hide_admin_msg_for_news(news)

    def __regenerate_news(self, news: GeneratedNews):
        NewsSubPage.objects.filter(pk=news.news_sub_page.pk).update(
            has_generated_news=True
        )

        self.__hide_news(news)

    @staticmethod
    def __hide_news(news: GeneratedNews):
        GeneratedNews.objects.filter(pk=news.pk).update(show_news=False)

    @staticmethod
    def __hide_admin_msg_for_news(news: GeneratedNews):
        GeneratedNews.objects.filter(pk=news.pk).update(show_admin_message=False)


class SummaryOfDayNewsController:
    def __init__(self):
        pass

    def get_summary_of_day(
        self, date: datetime.date, with_similarity: bool = False
    ) -> list:
        summaries = []
        for ssd in SingleDaySummary.objects.filter(day_to_summary=date):
            clusters = []
            clustering = ssd.clustering
            for cluster in Cluster.objects.filter(
                clustering=clustering.pk, is_outlier=False
            ):
                cl_data = ClusterSerializer(cluster, many=False).data
                if with_similarity:
                    cl_data["similarity"] = self.__get_similar_clusters(
                        cluster=cluster, as_data=True
                    )
                clusters.append(cl_data)
            res = {
                "info": SingleDaySummarySerializer(ssd, many=False).data,
                "clusters": clusters,
            }
            summaries.append(res)
        return summaries

    @staticmethod
    def __get_similar_clusters(cluster: Cluster, as_data: bool = True) -> dict:
        days_mapping = {}
        sc_cache_pk = {}
        for sc in SimilarClusters.objects.filter(source=cluster):
            # Get day of target cluster
            if sc.target.clustering.pk not in sc_cache_pk:
                for sd in SingleDaySummary.objects.filter(
                    clustering=sc.target.clustering
                ):
                    sc_cache_pk[sc.target.clustering.pk] = sd.day_to_summary
                    break

            t_day = sc_cache_pk.get(sc.target.clustering.pk, None)
            if t_day is None:
                continue

            if as_data:
                sc = SimilarClustersSerializer(sc, many=False).data

            t_day = t_day.strftime("%Y-%m-%d")
            if t_day not in days_mapping:
                days_mapping[t_day] = []
            days_mapping[t_day].append(sc)
        return days_mapping

    @staticmethod
    def get_summaries_for_day(date: datetime.date) -> QuerySet[SingleDaySummary]:
        return SingleDaySummary.objects.filter(day_to_summary=date)

    @staticmethod
    def force_drop_summary(summary: SingleDaySummary):
        # Drop clusters
        clusters = Cluster.objects.filter(clustering=summary.clustering.pk)
        print("  -> deleting clusters:", len(clusters))
        Cluster.objects.filter(clustering=summary.clustering.pk).delete()

        # Drop sample data
        print("Deep drop of summary object:", summary)
        for cluster in clusters:
            all_cld = SampleClusterData.objects.filter(pk=cluster.sample.pk)
            print("  -> deleting sample cluster data: ", len(all_cld))
            SampleClusterData.objects.filter(pk=cluster.sample.pk).delete()

        # Drop the single day summary
        summ = SingleDaySummary.objects.filter(pk=summary.pk)
        print("  -> deleting summary day:", summ)
        SingleDaySummary.objects.filter(pk=summary.pk).delete()

        # Drop clustering
        clustering = Clustering.objects.filter(pk=summary.clustering.pk)
        print("  -> deleting clustering:", clustering)
        Clustering.objects.filter(pk=summary.clustering.pk).delete()
