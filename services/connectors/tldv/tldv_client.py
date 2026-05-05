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


def get_meetings(since: datetime) -> list[dict]:
    api_key = os.environ["TLDV_API_KEY"]
    meetings = []
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

        for m in results:
            try:
                t = datetime.strptime(m.get("happenedAt", ""), _DATE_FMT).replace(tzinfo=timezone.utc)
                if t >= since:
                    meetings.append(m)
            except ValueError:
                pass

        if page >= total_pages:
            break

        if results:
            try:
                last_t = datetime.strptime(results[-1].get("happenedAt", ""), _DATE_FMT).replace(tzinfo=timezone.utc)
                if last_t < since:
                    break
            except ValueError:
                pass

        page += 1

    return meetings
