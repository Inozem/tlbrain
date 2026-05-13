# tlbrain


TLBrain is a cost-efficient memory system for Claude (Cowork), designed to retrieve structured knowledge from conversation transcripts.

The system helps Claude "remember" past client conversations through a lightweight retrieval layer powered by:

- TL;DV transcripts
- Google Drive storage
- Firestore state storage
- MCP remote server
- Dedicated sync/indexing service

TLBrain is optimized for **single-user / consultant workflows** with large volumes of client conversations and aims to be a cheaper alternative to traditional RAG stacks.

---

# Architecture

TLBrain uses a monorepo with two independent Cloud Run services, a Cloud Function checker, and a TL;DV connector.

## 1. MCP Service

Used by Claude / Cowork.

Responsibilities:

- remote MCP endpoint
- retrieval tools
- memory search
- user-facing requests

## 2. Vector Sync Service

Background indexing service.

Responsibilities:

- scan Google Drive folders
- read transcripts via Google Docs API
- parse utterances, generate summaries and structured facts via Gemini
- embed summaries and facts into Qdrant (vector store)
- detect content changes via SHA-256 hash
- maintain Firestore index with status machine (`queued → downloading → imported → syncing → synced / error`)
- recover stale syncing tasks automatically

## 3. Sync Checker (Cloud Function)

Lightweight scheduler triggered by Cloud Scheduler on a configurable interval (`SYNC_CHECKER_SCHEDULE`, default: every 15 minutes).

Responsibilities:

- detect new and modified documents via Drive Changes API (incremental, not full scan)
- mark changed documents as `imported` in Firestore
- dispatch per-document sync tasks to Cloud Tasks queue
- recover stale `syncing` and `downloading` documents automatically
- sync client registry from Drive folders

**Drive Changes API** replaces the previous full Drive scan — only actual changes are processed, making the checker faster and cheaper. The page token is stored in Firestore (`tokens/drive_sync`).

Intentionally deployed as a Cloud Function rather than Cloud Run to minimize cost — it runs on a fixed schedule with no idle traffic, so an always-on container would be wasteful.

## 4. TL;DV Connector

Three components that bring TL;DV transcripts into the system automatically.

### TL;DV Webhook (Cloud Function)

Receives `TranscriptReady` webhook from TL;DV, writes `status=queued` to Firestore, and dispatches an import task to Cloud Tasks. Idempotent.

### TL;DV Reconciliation (Cloud Function)

Runs daily at 03:00. Fetches TL;DV meetings from the last 48 hours, compares with Firestore, and dispatches import tasks for any missing transcripts. Idempotent.

### TL;DV Import Service (Cloud Run, scale to 0)

Receives an import task from Cloud Tasks with a `meeting_id`. Downloads the transcript from TL;DV API, detects the client name via a 3-stage chain (speaker frequency → Gemini by meeting name → Gemini by transcript excerpt → `_unassigned`), creates a native Google Doc in `ROOT_FOLDER/{client_name}/`, and writes `status=imported` to Firestore. The Vector Sync pipeline picks it up from there.

**Firestore status machine:**

```
queued → downloading → imported → syncing → synced
                                           ↘ error
```

---

# Repository Structure

```text
tlbrain/
├── core/
├── services/
│   ├── mcp/
│   ├── vector_sync/
│   ├── vector_sync_checker/
│   └── connectors/
│       └── tldv/
│           ├── import_service/
│           ├── reconciliation/
│           └── webhook/
├── infra/
│   ├── docker/
│   └── deploy/
└── README.md
```

---

# Quick Deploy Guide (Google Cloud)

## 1. Create Google Cloud Project

Open:

https://console.cloud.google.com/

Create a new project.

Recommended name:

`tlbrain-prod`

Copy your `PROJECT_ID`.

---

## 2. Install Google Cloud CLI

Install:

https://docs.cloud.google.com/sdk/docs/install-sdk

Then login:

```bash
gcloud auth login
```

Set project:

```bash
gcloud config set project YOUR_PROJECT_ID
```

---

## 3. Create Firestore Database

Open:

https://console.firebase.google.com/

Use your existing Google Cloud project.

Create Firestore database with:

- Edition: Standard
- Mode: Production
- Region: europe-west1
- Database ID: (default)

---

## 4. Configure Google Drive Folder

Recommended structure:

```text
Clients/
├── Client A/
│   └── meeting1.docx
├── Client B/
│   └── meeting2.docx
```

Copy folder URL and use it as `ROOT_FOLDER_URL`.

---

## 5. Deploy Services

```bash
bash infra/deploy/deploy.sh
```

This deploys:

- MCP service
- Sync service

---

## 6. Grant Google Drive Access to Cloud Run

Find sync service account:

```bash
gcloud run services describe tlbrain-sync \
--region europe-west1 \
--format="value(spec.template.spec.serviceAccountName)"
```

Share your Google Drive folder with this email.

Recommended permission: Viewer.

---

## 7. Configure `.env`

```env
# Version to deploy (must match a Docker Hub tag, e.g. v0.11)
VERSION=v0.11

# Google Cloud
PROJECT_ID=tlbrain-prod
REGION=europe-west1

# Service names
MCP_SERVICE_NAME=tlbrain-mcp
VECTOR_SYNC_SERVICE_NAME=tlbrain-vector-sync
SYNC_CHECKER_NAME=tlbrain-sync-checker

# Google Drive root folder URL
ROOT_FOLDER_URL=https://drive.google.com/drive/folders/YOUR_FOLDER_ID

# Gemini API
GEMINI_API_KEY=your-gemini-api-key

# Qdrant Cloud
QDRANT_URL=https://YOUR-CLUSTER-URL:6333
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=TLBrain

# Retrieval tuning
RETRIEVAL_TOP_K=15
RETRIEVAL_SCORE_THRESHOLD=0.6

# OAuth — restrict MCP access to a single Google account
ALLOWED_EMAIL=your-email@gmail.com

# Scheduler
VECTOR_SYNC_QUEUE=tlbrain-vector-sync-queue
CLOUD_TASKS_MAX_CONCURRENT=2
# How often Checker runs — controls update latency after Drive changes.
# "*/5 * * * *" = every 5 min (fast), "*/15 * * * *" = every 15 min (default), "0 4 * * *" = daily
SYNC_CHECKER_SCHEDULE="*/15 * * * *"

# TL;DV Connector
TLDV_API_KEY=your-tldv-api-key
TLDV_IMPORT_QUEUE=tlbrain-tldv-import-queue
TLDV_IMPORT_SERVICE_NAME=tlbrain-tldv-import
TLDV_WEBHOOK_FUNCTION_NAME=tlbrain-tldv-webhook
TLDV_RECONCILIATION_FUNCTION_NAME=tlbrain-tldv-reconciliation
# Daily reconciliation schedule (default: 03:00)
TLDV_RECONCILIATION_SCHEDULE="0 3 * * *"

# Local development only (Cloud Run uses ADC automatically)
# GOOGLE_APPLICATION_CREDENTIALS=./secrets/service-account.json
```

---

## 8. Redeploy After `.env` Changes

```bash
bash infra/deploy/deploy.sh
```

> After redeploying, remove the MCP server in Claude Cowork and add it again — the session expires on redeploy and needs to be re-authenticated.

---

## 9. Endpoints

### MCP

`https://YOUR-MCP-URL.run.app/mcp`

### Sync

Manual trigger:

```bash
curl -X POST https://YOUR-SYNC-URL.run.app/sync
```

Health check:

```bash
curl https://YOUR-SYNC-URL.run.app/
```

Sync response example:

```json
{
  "status": "ok",
  "result": {
    "files_found": 12,
    "imported": 3,
    "processed": 2,
    "skipped": 1,
    "deleted": 0,
    "recovered": 0,
    "errors": 0
  }
}
```

---

## 10. Set Up Google OAuth Client

One OAuth client is used for both MCP authentication and TL;DV Drive access.

1. Open [APIs & Services → Credentials](https://console.cloud.google.com/apis/credentials)
2. Click **Create Credentials → OAuth 2.0 Client ID**
3. Application type: **Web application**
4. Name: `TLBrain MCP`
5. Add both URIs to **Authorized redirect URIs**:
   - `https://claude.ai/api/mcp/auth_callback` — for Claude Cowork
   - `http://localhost:8085` — for local Drive token setup
6. Save and copy the **Client ID** and **Client Secret** into `.env`:

```env
GOOGLE_CLIENT_ID=your-client-id
GOOGLE_CLIENT_SECRET=your-client-secret
```

### Publish the OAuth App

1. Open [APIs & Services → Audience](https://console.cloud.google.com/auth/audience)
2. Click **Publish App** → confirm

This is required to avoid refresh tokens expiring every 7 days (Testing mode limitation).

> After publishing, Google will show an "This app isn't verified" warning on first authorization because Drive is a sensitive scope. Click **Advanced → Continue** to proceed — this is expected for self-hosted personal tools.

> If you skip publishing and stay in Testing mode, you must add your email under **Test users** and re-authorize every 7 days.

---

## 11. Set Up TL;DV Webhook

Deploy the TL;DV connector:

```bash
bash infra/deploy/connectors/deploy_tldv.sh
```

This deploys:

- TL;DV Import Service (Cloud Run)
- TL;DV Webhook Function (Cloud Function)
- TL;DV Reconciliation Function (Cloud Function + Cloud Scheduler)

After deploy, copy the webhook URL printed at the end and add it in TL;DV:

1. Open TL;DV → Settings → Integrations → Webhooks
2. Add webhook URL: `https://YOUR-WEBHOOK-URL`
3. Select event: `TranscriptReady`

From this point, every new TL;DV recording will automatically appear in the knowledge base within minutes. The daily reconciliation at 03:00 catches any missed webhooks.

---

## 12. Connect Claude Cowork to MCP

### Set up Google OAuth Client

Already done in step 10.

### Set ALLOWED_EMAIL

Add to `.env`:

```env
ALLOWED_EMAIL=your-email@gmail.com
```

Redeploy after this change.

### Connect in Claude Cowork

1. Open Claude Cowork → Settings → MCP Servers
2. Add server URL: `https://YOUR-MCP-URL.run.app/mcp`
3. Claude Cowork will detect OAuth automatically and prompt you to sign in with Google
4. Sign in with the same email as `ALLOWED_EMAIL`

> If `ALLOWED_EMAIL` is not set, the MCP endpoint is open without authentication.

---

## 13. Check Logs

```bash
gcloud run services logs read tlbrain-sync --region europe-west1 --limit 50
gcloud run services logs read tlbrain-mcp --region europe-west1 --limit 50
```

---

## 14. Set Up Qdrant Cloud

TLBrain uses Qdrant Cloud as the vector store for semantic search.

### Create a Free Cluster

Open:

https://cloud.qdrant.io

Sign up and create a free cluster:

- Cluster Name: `TLBrain`
- Cloud Provider: Google Cloud Platform
- Region: Frankfurt (europe-west3)
- Tier: Free (1 node, 4 GiB disk, 1 GiB RAM)

Click **Create Free Cluster**.

### Get Credentials

After the cluster is created:

1. Copy the **Cluster URL** (e.g. `https://ba911a96-xxxx.europe-west3.gcp.cloud.qdrant.io:6333`)
2. Go to **API Keys** → create a new key → copy it

### Verify Connection

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from core.qdrant.setup import ensure_collection
ensure_collection()
print('OK — collection ready')
"
```

---

# MCP Tools

| Tool | Description |
|---|---|
| `query` | Hybrid search (semantic + BM25 keyword) over client conversation transcripts. Supports `client_name`, `date_from`, `date_to` filters. |
| `get_transcript` | Retrieve full transcripts without semantic search. By `doc_id`, or by `client_name` with optional `limit` and date range. |
| `list_clients` | List all clients with dialog count and last dialog date. If unassigned transcripts exist, returns a `suggestion` and a `transcripts` list with `doc_id` and `dialog_date` to help assign them. |
| `create_client` | Create a new client: makes a folder in Google Drive and registers the client in the database. |
| `move_transcript` | Move a transcript to a different client folder. Updates Drive, resets the record for reindexing, removes old vectors, and enqueues sync immediately. |
| `import_all_transcripts` | Trigger a full import of all transcripts from TL;DV. Supports `limit` (default: 10) and `since` (ISO date) parameters. Idempotent. |
| `sync_changes` | Trigger an immediate sync run without waiting for the next scheduled check. Use after making changes in Drive. |
| `sync_status` | Show current sync status: counts by stage (queued, downloading, imported, syncing, synced, error) and unassigned transcript count. |

---

# Current Status

Implemented (v0.13):

- Hybrid retrieval in Stage 1 — parallel semantic search (dense, top-15, score ≥ 0.6) + BM25 keyword search (sparse, top-10) over utterances
- Named vectors in Qdrant collection — `dense` for summaries and facts, `bm25` sparse for utterances
- BM25 sparse vectors generated via fastembed (`Qdrant/bm25`) during indexing and at query time
- Keyword hits expanded to window `[i-2, i+2]` and merged with semantic covered ranges before fetch
- Default score threshold 0.6 for semantic hits (overridable via `RETRIEVAL_SCORE_THRESHOLD`)
- Retrieval logging — `query`, `semantic_hits`, `keyword_hits` counts per request

Implemented (v0.12):

- `create_client` MCP tool — creates Drive folder + Firestore record
- `move_transcript` MCP tool — moves file in Drive, updates Firestore, removes old vectors, enqueues sync immediately
- `import_all_transcripts` MCP tool — full TL;DV import with `limit` and `since` parameters
- `sync_changes` MCP tool — triggers immediate sync run on demand
- `sync_status` MCP tool — aggregated status counts from Firestore
- Drive Changes API in Checker — replaces full Drive scan with incremental change detection
- Speaker index in `clients` collection — enables fast client detection by speaker without scanning transcripts
- `setup_tokens.sh` — native OAuth setup without Docker or Python (PowerShell on Windows, Python3 on macOS/Linux)
- Qdrant collection auto-created on vector sync service startup

Implemented (v0.11):

- TL;DV Webhook Function — receives `TranscriptReady`, writes `queued` to Firestore, dispatches import task
- TL;DV Reconciliation Function — daily sync of last 48h, catches missed webhooks
- TL;DV Import Service — downloads transcript, detects client via 3-stage chain, creates Google Doc
- 3-stage client detection: speaker frequency → Gemini by meeting name → Gemini by transcript → `_unassigned`
- Firestore `queued` and `downloading` statuses — full state machine before `imported`
- `list_clients` — unassigned transcripts list with `doc_id` and `dialog_date`, `suggestion` field

Implemented (v0.10):

- monorepo architecture
- dual Cloud Run deployment
- MCP remote server with real semantic retrieval
- Google Drive connectivity via Docs API
- transcript parsing — utterances, metadata
- anchor-based windowing → summaries and facts via Gemini
- Gemini embeddings (768-dim)
- Qdrant Cloud vector store (utterances, summaries, facts)
- content hashing (sha256) — skip unchanged documents
- Firestore status machine (`imported → syncing → synced / error`)
- stale syncing recovery (auto-reset after 15 min)
- sync diff engine (new / updated / deleted)
- idempotent reindexing (append new → delete old versions)
- semantic search over summaries and facts (top-3 documents by score)
- retrieval pipeline: merge ranges → fetch utterances → dedup → segments
- each fact stored as individual Qdrant point with dedicated embedding
- enriched summary prompts (topic + specifics + decisions/next steps)
- score threshold filtering (`RETRIEVAL_SCORE_THRESHOLD`)
- `query` tool filters: `client_name`, `date_from`, `date_to`
- `get_transcript` tool — full transcripts by doc_id or client + date range
- `list_clients` tool — clients with dialog count and last dialog date
- retry with exponential backoff for transient external call failures
- structured JSON logging (Cloud Run / Cloud Logging compatible)
- combined summary + facts LLM call (single Gemini request per window)
- Cloud Tasks queue for parallel document sync
- Cloud Function checker — scans Drive on a configurable schedule (default: every 15 min), dispatches sync tasks via Cloud Tasks
- Google OAuth 2.0 for MCP endpoint (single-user access via ALLOWED_EMAIL)

---

# Roadmap

- ~~v0.3 — Gemini Memory Layer (utterances, summaries, structured facts)~~ ✓
- ~~v0.4 — Retrieval Validation~~ ✓
- ~~v0.5 — MCP Real Retrieval~~ ✓
- ~~v0.6 — Retrieval Quality + Facts~~ ✓
- ~~v0.7 — Full Retrieval Pipeline~~ ✓
- ~~v0.8 — Filters + get_transcript + list_clients~~ ✓
- ~~v0.9 — Scheduler + Stability~~ ✓
- ~~v0.10 — Production Ready~~ ✓
- ~~v0.11 — TL;DV Connector~~ ✓
- ~~v0.12 — MCP Management Tools~~ ✓
- ~~v0.13 — Hybrid Search (BM25 + Semantic)~~ ✓

---

# Vision

TLBrain aims to become a personal memory layer for Claude — focused, cheap, private, and practical.


