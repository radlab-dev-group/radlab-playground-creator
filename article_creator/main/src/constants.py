import os
from django.conf import settings

STATIC_DIR = "static/"
CONFIG_DIR = "configs/"

SYSTEM_CONFIG_FILENAME = "django-config.json"
KEYCLOAK_CONFIG_FILENAME = "keycloak-config.json"
SYSTEM_CONFIG_FILENAME_DEFAULT = "django-config-default.json"

SECRET_KEY_FILENAME = "secret-key.txt"
SECRET_KEY_FILENAME_DEFAULT = "secret-key-default.txt"

AVAILABLE_CONFIG_DIRS = [
    STATIC_DIR,
    CONFIG_DIR,
    os.path.join(STATIC_DIR, CONFIG_DIR),
]

AVAILABLE_LANGUAGES = ["pl", "en"]

ENV_DEBUG = "ENV_DEBUG"
ENV_SHOW_ADMIN_WINDOW = "ENV_SHOW_ADMIN_WINDOW"
ENV_PUBLIC_API_AVAILABLE = "ENV_PUBLIC_API_AVAILABLE"
ENV_DEFAULT_LANGUAGE = "ENV_DEFAULT_LANGUAGE"
ENV_SECRET_KEY = "ENV_SECRET_KEY"
ENV_ALLOWED_HOSTS = "ENV_ALLOWED_HOSTS"
ENV_INSTALLED_APPS = "ENV_INSTALLED_APPS"

ENV_DB_NAME = "ENV_DB_NAME"
ENV_DB_HOST = "ENV_DB_HOST"
ENV_DB_PORT = "ENV_DB_PORT"
ENV_DB_ENGINE = "ENV_DB_ENGINE"
ENV_DB_USERNAME = "ENV_DB_USERNAME"
ENV_DB_PASSWORD = "ENV_DB_PASSWORD"

ENV_LOGGER = "ENV_LOGGER"

ENV_EXTERNAL_API = "ENV_EXTERNAL_API"

ENV_USE_KC_AUTH = "ENV_USE_KC_AUTH"


def default_app_language():
    """
    Returns default application language
    :return: String-like application language
    """
    return settings.DEFAULT_APP_LANGUAGE


def get_logger(name: str = None):
    """
    Returns logger defined into settings as global logger
    :param name: Optionally specify logger name
    :return: Built logger object
    """
    return settings.MAIN_LOGGER


def prepare_api_url(endpoint_url: str) -> str:
    """
    Based on main_api_endpoint url, prepare whole api url with (optionally) version
    :param endpoint_url: Endpoint url (without root call, only ep name)
    :return: Prepared whole api url
    """
    main_url = settings.MAIN_API_URL
    if main_url.endswith("/"):
        main_url = main_url[:-1]

    endpoint_url = endpoint_url.strip()
    if endpoint_url.startswith("/"):
        endpoint_url = endpoint_url[1:]
    whole_url = f"{main_url}/{endpoint_url}"

    get_logger().info(f"Sharing api url {whole_url}")

    return whole_url
