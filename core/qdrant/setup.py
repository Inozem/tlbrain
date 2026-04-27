from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from core.qdrant.client import get_client
from core.qdrant.schema import COLLECTION_CONFIG, COLLECTION_NAME

_KEYWORD_INDEXES = ["doc_id", "version", "root_folder_id", "type", "client_name", "dialog_date"]


def ensure_collection() -> None:
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=COLLECTION_CONFIG["size"],
                distance=Distance.COSINE,
                on_disk=COLLECTION_CONFIG["on_disk"],
            ),
        )
    _ensure_indexes(client)


def _ensure_indexes(client) -> None:
    info = client.get_collection(COLLECTION_NAME)
    existing = set(info.payload_schema.keys()) if info.payload_schema else set()
    for field in _KEYWORD_INDEXES:
        if field not in existing:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
