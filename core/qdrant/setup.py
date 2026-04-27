from qdrant_client.models import Distance, VectorParams
from core.qdrant.client import get_client
from core.qdrant.schema import COLLECTION_NAME, COLLECTION_CONFIG


def ensure_collection() -> None:
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=COLLECTION_CONFIG["size"],
            distance=Distance.COSINE,
            on_disk=COLLECTION_CONFIG["on_disk"],
        ),
    )
