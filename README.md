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

## 3. Enable Required Services

```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable drive.googleapis.com
gcloud services enable firestore.googleapis.com
```

---

## 4. Create Firestore Database

Open:

https://console.firebase.google.com/

Use your existing Google Cloud project.

Create Firestore database with:

- Edition: Standard
- Mode: Production
- Region: europe-west1
- Database ID: (default)

---

## 5. Configure Google Drive Folder

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

## 6. Deploy Services

```bash
bash infra/deploy/deploy.sh
```

This deploys:

- MCP service
- Sync service

---

## 7. Grant Google Drive Access to Cloud Run

Find sync service account:

```bash
gcloud run services describe tlbrain-sync \
--region europe-west1 \
--format="value(spec.template.spec.serviceAccountName)"
```

Share your Google Drive folder with this email.

Recommended permission: Viewer.

---

## 8. Configure `.env`

```env
PROJECT_ID=tlbrain-prod
REGION=europe-west1

MCP_SERVICE_NAME=tlbrain-mcp
SYNC_SERVICE_NAME=tlbrain-sync

ROOT_FOLDER_URL=https://drive.google.com/drive/folders/YOUR_FOLDER_ID
```

---

## 9. Redeploy After `.env` Changes

```bash
bash infra/deploy/deploy.sh
```

---

## 10. Endpoints

### MCP

`https://YOUR-MCP-URL.run.app/mcp`

### Sync

Manual trigger:

`POST https://YOUR-SYNC-URL.run.app/sync`

Health check:

`GET https://YOUR-SYNC-URL.run.app/`

---

# Current Status

Implemented:

- monorepo architecture
- dual Cloud Run deployment
- MCP remote server
- sync service foundation
- Google Drive connectivity
- Firestore index storage

---

# Roadmap

Planned next:

- `.docx` transcript parsing
- real content hashing
- vector indexing
- semantic retrieval improvements
- production scheduler

---

# Vision

TLBrain aims to become a personal memory layer for Claude — focused, cheap, private, and practical.


