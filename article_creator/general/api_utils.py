import os
import requests

PUBLIC_USER_NAME = os.getenv("PLAYGROUND_PUBLIC_USER_NAME", None)
PUBLIC_USER_PASS = os.getenv("PLAYGROUND_PUBLIC_USER_PASS", None)


class BasePublicApiInterface:

    @staticmethod
    def general_call_get(
        host_url: str | None,
        endpoint: str,
        params: dict | None = None,
        data: dict | None = None,
        json_data: dict | None = None,
        headers: dict | None = None,
        login_url: str | None = None,
    ):
        user_api_call_url, headers = (
            BasePublicApiInterface.__prepare_api_url_and_headers(
                host_url=host_url,
                endpoint=endpoint,
                login_url=login_url,
                headers=headers,
            )
        )
        response = requests.get(
            user_api_call_url,
            params=params,
            data=data,
            json=json_data,
            headers=headers,
        )
        if response.ok:
            return response.json()
        return response

    @staticmethod
    def general_call_post(
        host_url: str | None,
        endpoint: str,
        params: dict | None = None,
        data: dict | None = None,
        files=None,
        json_data: dict | None = None,
        headers: dict | None = None,
        login_url: str | None = None,
    ):
        user_api_call_url, headers = (
            BasePublicApiInterface.__prepare_api_url_and_headers(
                host_url=host_url,
                endpoint=endpoint,
                login_url=login_url,
                headers=headers,
            )
        )
        response = requests.post(
            user_api_call_url,
            params=params,
            files=files,
            data=data,
            json=json_data,
            headers=headers,
        )
        if response.ok:
            return response.json()
        return response

    @staticmethod
    def login(login_url: str, username: str, password: str) -> str | None:
        context = {"username": username, "password": password}
        response = requests.post(login_url, data=context)
        if "token" in response.json():
            return response.json()["token"]
        return None

    @staticmethod
    def __prepare_api_url_and_headers(host_url, endpoint, login_url, headers):
        BasePublicApiInterface.__assert_public_variables()

        if host_url is None or not len(host_url):
            user_api_call_url = endpoint
        else:
            user_api_call_url = "{}/{}".format(
                host_url.strip("/"), endpoint.strip("/")
            )

        if login_url is not None and len(login_url):
            user_token = BasePublicApiInterface.login(
                login_url=login_url,
                username=PUBLIC_USER_NAME,
                password=PUBLIC_USER_PASS,
            )
            if headers is None:
                headers = {}
            headers["Authorization"] = f"Token {user_token}"

        return user_api_call_url, headers

    @staticmethod
    def __assert_public_variables():
        if PUBLIC_USER_NAME is None:
            raise ValueError("PLAYGROUND_PUBLIC_USER_NAME must be defined")
        if PUBLIC_USER_PASS is None:
            raise ValueError("PLAYGROUND_PUBLIC_USER_NAME must be defined")
