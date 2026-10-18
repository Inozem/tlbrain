"""Google Docs document creation and text styling operations."""


def create_google_doc(
    drive,
    docs,
    folder_id: str,
    title: str,
    text: str,
    speaker_ranges: list[tuple[int, int]],
) -> tuple[str, str]:
    """Create a native Google Doc and insert ready-to-write transcript text."""
    file_meta = drive.files().create(
        body={
            "name": title,
            "mimeType": "application/vnd.google-apps.document",
            "parents": [folder_id],
        },
        fields="id,modifiedTime",
    ).execute()
    doc_id = file_meta["id"]
    modified_time = file_meta.get("modifiedTime", "")

    requests_body = [{"insertText": {"location": {"index": 1}, "text": text}}]
    for start, end in speaker_ranges:
        requests_body.append({
            "updateTextStyle": {
                "range": {"startIndex": start + 1, "endIndex": end + 1},
                "textStyle": {"bold": True},
                "fields": "bold",
            }
        })

    docs.documents().batchUpdate(
        documentId=doc_id, body={"requests": requests_body}
    ).execute()
    return doc_id, modified_time
