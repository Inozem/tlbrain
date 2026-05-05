import os
from datetime import datetime, timezone

import google.api_core.exceptions
import google.auth
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google.cloud import firestore
from googleapiclient.discovery import build

from core.config import get_root_folder_id
from core.google_drive.firestore import COLLECTION_NAME
from core.utils.tasks import enqueue_task
from tldv_client import tldv_get

_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
]

app = FastAPI()


def _build_credentials():
    refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN")
    if refresh_token:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            client_id=os.environ["GOOGLE_CLIENT_ID"],
            client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
            token_uri="https://oauth2.googleapis.com/token",
        )
        creds.refresh(Request())
        return creds
    creds, _ = google.auth.default(scopes=_SCOPES)
    return creds


def _build_drive():
    return build("drive", "v3", credentials=_build_credentials())


def _build_docs():
    return build("docs", "v1", credentials=_build_credentials())


def _get_or_create_folder(drive, parent_id: str, name: str) -> str:
    results = drive.files().list(
        q=f"name={name!r} and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false",
        fields="files(id)",
        pageSize=1,
    ).execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]
    folder = drive.files().create(
        body={
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        },
        fields="id",
    ).execute()
    return folder["id"]


def _parse_happened_at(happened_at: str) -> tuple[str, str]:
    if not happened_at:
        return "", ""
    try:
        dt = datetime.fromisoformat(happened_at.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
    except ValueError:
        return "", ""


def _format_transcript(meeting: dict, utterances: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    source_file = meeting.get("name") or f"TL;DV {meeting.get('id', '')}"
    dialog_date, dialog_time = _parse_happened_at(meeting.get("happenedAt", ""))

    header = "\n".join([
        f"DATE: {dialog_date}",
        f"TIME: {dialog_time}",
        "PROVIDER: tldv",
        f"SOURCE_FILE: {source_file}",
        "---",
    ]) + "\n"

    speaker_ranges: list[tuple[int, int]] = []
    body_parts: list[str] = []
    pos = len(header)

    for u in utterances:
        speaker = (u.get("speaker") or u.get("speakerName", "")).replace("::", "")
        text = u.get("text") or u.get("transcript", "")
        if not text:
            continue
        if speaker:
            speaker_ranges.append((pos, pos + len(speaker)))
            line = f"{speaker} :: {text}"
        else:
            line = text
        body_parts.append(line)
        pos += len(line) + 2  # +2 for \n\n separator

    return header + "\n\n".join(body_parts), speaker_ranges


def _create_google_doc(drive, folder_id: str, title: str, text: str, speaker_ranges: list[tuple[int, int]]) -> tuple[str, str]:
    docs = _build_docs()

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

    docs.documents().batchUpdate(documentId=doc_id, body={"requests": requests_body}).execute()

    return doc_id, modified_time


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/import")
async def import_meeting(request: Request):
    body = await request.json()
    meeting_id = body.get("meeting_id")
    if not meeting_id:
        return JSONResponse({"error": "missing meeting_id"}, status_code=400)

    print(f"Importing meeting_id={meeting_id}", flush=True)

    db = firestore.Client()
    existing = list(
        db.collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("tldv_meeting_id", "==", meeting_id))
        .limit(1)
        .stream()
    )
    if existing:
        print(f"Already imported: {meeting_id}", flush=True)
        return {"ok": True, "skipped": True}

    meeting = tldv_get(f"/meetings/{meeting_id}")
    print(f"Meeting: {meeting.get('name')!r}", flush=True)

    transcript_resp = tldv_get(f"/meetings/{meeting_id}/transcript")
    utterances = transcript_resp.get("results", transcript_resp.get("data", []))
    print(f"Utterances: {len(utterances)}", flush=True)

    if not utterances:
        print(f"No transcript yet for meeting_id={meeting_id}, skipping", flush=True)
        return {"ok": True, "skipped": True, "reason": "no_transcript"}

    client_name = os.environ.get("TLDV_CLIENT_NAME", "_unassigned")
    title = meeting.get("name") or f"TL;DV {meeting_id}"
    text, speaker_ranges = _format_transcript(meeting, utterances)
    dialog_date, _ = _parse_happened_at(meeting.get("happenedAt", ""))

    root_folder_id = get_root_folder_id()
    drive = _build_drive()
    client_folder_id = _get_or_create_folder(drive, root_folder_id, client_name)

    doc_id, modified_time = _create_google_doc(drive, client_folder_id, title, text, speaker_ranges)
    print(f"Created Google Doc: doc_id={doc_id} title={title!r}", flush=True)

    db.collection(COLLECTION_NAME).document(doc_id).set({
        "doc_id": doc_id,
        "root_folder_id": root_folder_id,
        "client_name": client_name,
        "drive_folder": client_folder_id,
        "dialog_date": dialog_date,
        "provider": "tldv",
        "source_file": title,
        "tldv_meeting_id": meeting_id,
        "modifiedTime": modified_time,
        "status": "imported",
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "syncing_started_at": None,
        "synced_at": None,
        "error": None,
    })
    print(f"Saved to Firestore: doc_id={doc_id}", flush=True)

    vector_sync_url = os.environ.get("VECTOR_SYNC_URL", "")
    vector_sync_queue = os.environ.get("VECTOR_SYNC_QUEUE", "")

    if vector_sync_url and vector_sync_queue:
        enqueue_task(
            queue_name=vector_sync_queue,
            task_id=doc_id,
            url=f"{vector_sync_url}/sync/doc/{doc_id}",
        )
        print(f"Queued vector sync: doc_id={doc_id}", flush=True)
    else:
        print("VECTOR_SYNC_URL or VECTOR_SYNC_QUEUE not set, skipping sync queue", flush=True)

    return {"ok": True, "doc_id": doc_id, "meeting_id": meeting_id}
