import os
import re


def get_root_folder_id() -> str:
    url = os.environ["ROOT_FOLDER_URL"]
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if not match:
        raise ValueError(f"Cannot extract folder ID from ROOT_FOLDER_URL: {url}")
    return match.group(1)


def get_retrieval_top_k() -> int:
    return int(os.environ.get("RETRIEVAL_TOP_K", "15"))


def get_retrieval_score_threshold() -> float:
    return float(os.environ.get("RETRIEVAL_SCORE_THRESHOLD", "0.6"))
