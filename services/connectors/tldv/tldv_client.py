import os
from datetime import datetime, timezone

import requests

_BASE = "https://pasta.tldv.io/v1alpha1"
_DATE_FMT = "%a %b %d %Y %H:%M:%S GMT+0000 (Coordinated Universal Time)"


def tldv_get(path: str) -> dict:
    resp = requests.get(
        f"{_BASE}{path}",
        headers={"x-api-key": os.environ["TLDV_API_KEY"]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def iter_meeting_pages(since: datetime | None = None):
    """Yield meetings one page at a time from TL;DV API."""
    api_key = os.environ["TLDV_API_KEY"]
    page = 1

    while True:
        resp = requests.get(
            f"{_BASE}/meetings",
            headers={"x-api-key": api_key},
            params={"page": page},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        total_pages = data.get("pages", 1)

        if since is None:
            yield results
        else:
            filtered = []
            stop = False
            for m in results:
                try:
                    t = datetime.strptime(m.get("happenedAt", ""), _DATE_FMT).replace(tzinfo=timezone.utc)
                    if t >= since:
                        filtered.append(m)
                    else:
                        stop = True
                except ValueError:
                    pass
            yield filtered
            if stop:
                return

        if page >= total_pages:
            break

        page += 1


def get_meetings(since: datetime | None = None) -> list[dict]:
    """Fetch all meetings from TL;DV API. If since is None, returns all meetings."""
    return [m for page in iter_meeting_pages(since) for m in page]
