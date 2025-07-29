"""ECB proxies to connect with external networks."""

from __future__ import annotations

import os
import platform
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, ParamSpec, TypeVar

# import ecb_certifi
import requests
from huggingface_hub import configure_http_backend

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

WIN_PROXY = "http://ap-python-proxy:x2o7rCPYuN1JuV8H@app-gw-2.ecb.de:8080"
AML_PROXY = "http://10.141.0.165:3128"
HF_ENDPOINT = (
    "https://artifactory.sofa.dev/artifactory/api/huggingfaceml/huggingface-remote"
)


def can_connect_to_google() -> bool:
    try:
        response = requests.get("https://www.google.com", timeout=1)
        response.raise_for_status()
    except requests.RequestException:
        return False
    return True


def get_proxy() -> str:
    if platform.system() == "Windows":
        return WIN_PROXY

    if os.getenv("AML_IS_SECURED_WS") or not can_connect_to_google():
        return AML_PROXY

    return ""


def get_ssl() -> str:
    return False  # ecb_certifi.where()


@contextmanager
def set_proxies(*, set_ssl: bool = True) -> Generator[None, None, None]:
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "REQUESTS_CA_BUNDLE")
    old_env = {k: os.getenv(k) for k in keys}
    proxy = get_proxy()

    try:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        if set_ssl:
            os.environ["REQUESTS_CA_BUNDLE"] = get_ssl()
        yield
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


P = ParamSpec("P")
R = TypeVar("R")


def with_proxies(
    func: Callable[P, R] | None = None, *, set_ssl: bool = True
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with set_proxies(set_ssl=set_ssl):
                return f(*args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def setup_huggingface():
    # -- To be deprecated since artifactory support got released ----------------
    def backend_factory() -> requests.Session:
        custom_session = requests.Session()
        custom_session.proxies = {
            "http": get_proxy(),
            "https": get_proxy(),
        }
        custom_session.verify = get_ssl()
        return custom_session

    configure_http_backend(backend_factory=backend_factory)
    # -- To be deprecated since artifactory support got released ----------------

    # -- NOT YET STABLE
    # -- https://sofa.pages.sofa.dev/for-users/sofa-wiki/huggingface-artifactory/
    # os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    # os.environ["TMPDIR"] = "./temporary_cache"
    # os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
    # os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
