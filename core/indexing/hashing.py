import hashlib


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_utterance(order_index: int, speaker: str, text: str) -> str:
    return sha256_text(f"{order_index}:{speaker}:{text}")
