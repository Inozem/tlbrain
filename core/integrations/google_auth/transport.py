"""One-request httplib2-compatible transport for trusted Telegram reads."""

from __future__ import annotations

import httplib2
import requests
from requests.adapters import HTTPAdapter


class SingleAttemptHttp(httplib2.Http):
    """Use requests with adapter retries and redirects disabled for one RPC."""

    def __init__(self, *, timeout: float = 60) -> None:
        super().__init__(timeout=timeout)
        self._session = requests.Session()
        adapter = HTTPAdapter(max_retries=0)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def request(
        self, uri, method="GET", body=None, headers=None,
        redirections=httplib2.DEFAULT_MAX_REDIRECTS, connection_type=None, **kwargs,
    ):
        del redirections, connection_type
        response = self._session.request(
            method=method, url=uri, data=body, headers=headers,
            timeout=kwargs.pop("timeout", self.timeout), allow_redirects=False,
            **kwargs,
        )
        metadata = {key.lower(): value for key, value in response.headers.items()}
        metadata.update({"status": str(response.status_code), "reason": response.reason})
        return httplib2.Response(metadata), response.content

    def close(self) -> None:
        self._session.close()
