import os
import re
from datetime import datetime, timezone

import google.api_core.exceptions
import google.auth
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google.cloud import firestore, tasks_v2
from googleapiclient.discovery import build

_TLDV_API_BASE = "https://pasta.tldv.io/v1alpha1"
_COLLECTION = "transcript_index"
_CLIENT_NAME = "tldv"
_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
]

app = FastAPI()


def _get_root_folder_id() -> str:
    url = os.environ["ROOT_FOLDER_URL"]
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if not match:
        raise ValueError(f"Cannot extract folder ID from ROOT_FOLDER_URL: {url}")
    return match.group(1)


def _tldv_get(path: str) -> dict:
    api_key = os.environ["TLDV_API_KEY"]
    resp = requests.get(
        f"{_TLDV_API_BASE}{path}",
        headers={"x-api-key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


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


def _format_transcript(meeting: dict, utterances: list[dict]) -> str:
    source_file = meeting.get("name") or f"TL;DV {meeting.get('id', '')}"
    dialog_date, dialog_time = _parse_happened_at(meeting.get("happenedAt", ""))

    lines = [
        f"DATE: {dialog_date}",
        f"TIME: {dialog_time}",
        "PROVIDER: tldv",
        f"SOURCE_FILE: {source_file}",
        "---",
    ]
    for u in utterances:
        speaker = (u.get("speaker") or u.get("speakerName", "")).replace("::", "")
        text = u.get("text") or u.get("transcript", "")
        if text:
            lines.append(f"{speaker} :: {text}" if speaker else text)

    return "\n".join(lines)


def _create_google_doc(drive, folder_id: str, title: str, text: str) -> tuple[str, str]:
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

    docs.documents().batchUpdate(
        documentId=doc_id,
        body={"requests": [{"insertText": {"location": {"index": 1}, "text": text}}]},
    ).execute()

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
        db.collection(_COLLECTION)
        .where(filter=firestore.FieldFilter("tldv_meeting_id", "==", meeting_id))
        .limit(1)
        .stream()
    )
    if existing:
        print(f"Already imported: {meeting_id}", flush=True)
        return {"ok": True, "skipped": True}

    meeting = _tldv_get(f"/meetings/{meeting_id}")
    print(f"Meeting: {meeting.get('name')!r}", flush=True)

    transcript_resp = _tldv_get(f"/meetings/{meeting_id}/transcript")
    utterances = transcript_resp.get("results", transcript_resp.get("data", []))
    print(f"Utterances: {len(utterances)}", flush=True)

    title = meeting.get("name") or f"TL;DV {meeting_id}"
    text = _format_transcript(meeting, utterances)
    dialog_date, _ = _parse_happened_at(meeting.get("happenedAt", ""))

    root_folder_id = _get_root_folder_id()
    drive = _build_drive()
    client_folder_id = _get_or_create_folder(drive, root_folder_id, _CLIENT_NAME)

    doc_id, modified_time = _create_google_doc(drive, client_folder_id, title, text)
    print(f"Created Google Doc: doc_id={doc_id} title={title!r}", flush=True)

    db.collection(_COLLECTION).document(doc_id).set({
        "doc_id": doc_id,
        "root_folder_id": root_folder_id,
        "client_name": _CLIENT_NAME,
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
    region = os.environ["REGION"]
    _, project_id = google.auth.default()
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", project_id)

    if vector_sync_url and vector_sync_queue:
        tasks_client = tasks_v2.CloudTasksClient()
        queue_path = tasks_client.queue_path(project_id, region, vector_sync_queue)
        task = {
            "name": tasks_client.task_path(project_id, region, vector_sync_queue, doc_id),
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"{vector_sync_url}/sync/doc/{doc_id}",
            },
        }
        try:
            tasks_client.create_task(parent=queue_path, task=task)
            print(f"Queued vector sync: doc_id={doc_id}", flush=True)
        except google.api_core.exceptions.AlreadyExists:
            print(f"Vector sync task already exists: doc_id={doc_id}", flush=True)
    else:
        print("VECTOR_SYNC_URL or VECTOR_SYNC_QUEUE not set, skipping sync queue", flush=True)

    return {"ok": True, "doc_id": doc_id, "meeting_id": meeting_id}
