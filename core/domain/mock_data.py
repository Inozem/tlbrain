from services.mcp.app.mcp.schemas import TLBrainMeta


def get_mock_segments() -> list[dict]:
    return [
        {
            "doc_id": "doc_1",
            "client_name": "Client A",
            "dialog_date": "2024-01-10",
            "range": [1, 3],
            "dialog": [
                {
                    "speaker": "Client",
                    "text": "We are interested in pricing",
                    "timestamp": "00:01",
                    "order_index": 1,
                },
                {
                    "speaker": "Manager",
                    "text": "Our pricing starts from $100",
                    "timestamp": "00:02",
                    "order_index": 2,
                },
                {
                    "speaker": "Client",
                    "text": "Can we get a discount?",
                    "timestamp": "00:03",
                    "order_index": 3,
                },
            ],
        }
    ]


def query_handler(query: str) -> tuple[list[dict], TLBrainMeta]:
    segments = get_mock_segments()

    meta = TLBrainMeta(
        truncated=False,
        total_matches=len(segments),
        returned_segments=len(segments),
        limit_reason=None,
        suggestion=None,
    )

    return segments, meta
