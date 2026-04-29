from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from core.qdrant.client import get_client
from core.qdrant.schema import COLLECTION_CONFIG, get_collection_name

_KEYWORD_INDEXES = ["doc_id", "version", "root_folder_id", "type", "client_name", "dialog_date"]
_INTEGER_INDEXES = ["order_index", "dialog_date_num"]


def ensure_collection() -> None:
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    if get_collection_name() not in existing:
        client.create_collection(
            collection_name=get_collection_name(),
            vectors_config=VectorParams(
                size=COLLECTION_CONFIG["size"],
                distance=Distance.COSINE,
                on_disk=COLLECTION_CONFIG["on_disk"],
            ),
        )
    _ensure_indexes(client)


def _ensure_indexes(client) -> None:
    info = client.get_collection(get_collection_name())
    existing = set(info.payload_schema.keys()) if info.payload_schema else set()
    for field in _KEYWORD_INDEXES:
        if field not in existing:
            client.create_payload_index(
                collection_name=get_collection_name(),
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
    for field in _INTEGER_INDEXES:
        if field not in existing:
            client.create_payload_index(
                collection_name=get_collection_name(),
                field_name=field,
                field_schema=PayloadSchemaType.INTEGER,
            )
