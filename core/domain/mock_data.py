def get_mock_segments():
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
                    "order_index": 1
                },
                {
                    "speaker": "Manager",
                    "text": "Our pricing starts from $100",
                    "timestamp": "00:02",
                    "order_index": 2
                },
                {
                    "speaker": "Client",
                    "text": "Can we get a discount?",
                    "timestamp": "00:03",
                    "order_index": 3
                }
            ]
        }
    ]
