import os

EMBEDDING_DIMENSIONS = 768

COLLECTION_CONFIG = {
    "size": EMBEDDING_DIMENSIONS,
    "on_disk": True,
}


def get_collection_name() -> str:
    return os.environ["QDRANT_COLLECTION"]
