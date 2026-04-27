from typing import Any


def generate_windows(
    utterances: list[dict[str, Any]],
    anchor_step: int = 3,
    half_window: int = 2,
) -> list[dict[str, Any]]:
    """
    For each anchor utterance (every anchor_step-th), produce a context window
    of [i - half_window .. i + half_window] utterances.

    Returns a list of window dicts:
      center_index  — order_index of the anchor utterance
      covered_range — [first_order_index, last_order_index] in the window
      utterances    — the utterance dicts inside the window
    """
    if not utterances:
        return []

    windows = []
    n = len(utterances)

    for i in range(0, n, anchor_step):
        start = max(0, i - half_window)
        end = min(n - 1, i + half_window)
        window_utterances = utterances[start : end + 1]
        windows.append({
            "center_index": utterances[i]["order_index"],
            "covered_range": [
                utterances[start]["order_index"],
                utterances[end]["order_index"],
            ],
            "utterances": window_utterances,
        })

    return windows
