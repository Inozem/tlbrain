# tlbrain

> 🚧 This project is under active development.

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

TLBrain uses a monorepo with two independent Cloud Run services.

## 1. MCP Service

Used by Claude / Cowork.

Responsibilities:

- remote MCP endpoint
- retrieval tools
- memory search
- user-facing requests

## 2. Sync Service

Background indexing service.

Responsibilities:

- scan Google Drive folders
- parse `.docx` transcripts
- detect changes
- sync searchable metadata
- maintain Firestore index

---

# Repository Structure

```text
tlbrain/
├── core/
├── services/
│   ├── mcp/
│   └── sync/
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
# Google Cloud
PROJECT_ID=tlbrain-prod
REGION=europe-west1

# Service names
MCP_SERVICE_NAME=tlbrain-mcp
SYNC_SERVICE_NAME=tlbrain-sync

# Google Drive root folder URL
ROOT_FOLDER_URL=https://drive.google.com/drive/folders/YOUR_FOLDER_ID

# Runtime (optional, defaults shown)
PORT=8080
ENVIRONMENT=dev

# Local development only (Cloud Run uses ADC automatically)
# GOOGLE_APPLICATION_CREDENTIALS=./secrets/service-account.json
```

---

## 8. Redeploy After `.env` Changes

```bash
bash infra/deploy/deploy.sh
```

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
    "new": 2,
    "updated": 1,
    "skipped": 8,
    "deleted": 0,
    "ignored": 1
  }
}
```

---

## 10. Check Logs

```bash
gcloud run services logs read tlbrain-sync --region europe-west1 --limit 50
gcloud run services logs read tlbrain-mcp --region europe-west1 --limit 50
```

---

## 11. Set Up Qdrant Cloud (v0.3+)

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

### Add to `.env`

```env
QDRANT_URL=https://YOUR-CLUSTER-URL:6333
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION=TLBrain
```

### Verify Connection

```bash
python -c "
from dotenv import load_dotenv
load_dotenv()
from core.qdrant.setup import ensure_collection
ensure_collection()
print('OK — collection created or already exists')
"
```

---

# Current Status

Implemented (v0.2):

- monorepo architecture
- dual Cloud Run deployment
- MCP remote server (mock response)
- Google Drive connectivity
- `.docx` transcript parsing
- content hashing (sha256)
- Firestore index storage
- sync diff engine (new / updated / deleted)
- sync reporting and safe file filters

---

# Roadmap

- v0.3 — Gemini Memory Layer (utterances, summaries, structured facts)
- v0.4 — Retrieval Validation
- v0.5 — MCP Real Retrieval
- v0.6 — Retrieval Quality + Facts
- v0.7 — Full Retrieval Pipeline
- v0.8 — Filters (client, date range)
- v0.9 — Scheduler + Stability
- v1.0 — Production MVP

---

# Vision

TLBrain aims to become a personal memory layer for Claude — focused, cheap, private, and practical.


