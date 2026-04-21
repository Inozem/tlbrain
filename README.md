# tlbrain

> 🚧 This project is under active development.

TLBrain is a cost-efficient memory system for Claude (Cowork), designed to retrieve structured knowledge from conversation transcripts.

The system helps Claude "remember" past client conversations through a lightweight retrieval layer powered by:

* TL;DV transcripts
* Google Drive storage
* MCP remote server
* Dedicated sync/indexing service

TLBrain is optimized for **single-user / consultant workflows** with large volumes of client conversations and aims to be a cheaper alternative to traditional RAG stacks.

---

# Architecture

TLBrain uses a monorepo with two independent Cloud Run services:

## 1. MCP Service

Used by Claude / Cowork.

Responsibilities:

* remote MCP endpoint
* retrieval tools
* memory search
* user-facing requests

## 2. Sync Service

Background indexing service.

Responsibilities:

* scan Google Drive folders
* parse `.docx` transcripts
* sync new / changed files
* prepare searchable memory data

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

```text
tlbrain-prod
```

Copy your:

```text
PROJECT_ID
```

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

Run:

```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable drive.googleapis.com
```

---

## 4. Configure Google Drive Folder

Create a Google Drive folder for transcripts.

Recommended structure:

```text
Clients/
├── Client A/
│   └── meeting1.docx
├── Client B/
│   └── meeting2.docx
```

Copy folder URL:

```text
https://drive.google.com/drive/folders/YOUR_FOLDER_ID
```

---

## 5. Deploy Services

Run:

```bash
bash infra/deploy/deploy.sh
```

This deploys:

* MCP service
* Sync service

---

## 6. Grant Google Drive Access to Cloud Run

After first deploy, Cloud Run uses a service account identity.

Find sync service account:

```bash
gcloud run services describe tlbrain-sync \
--region europe-west1 \
--format="value(spec.template.spec.serviceAccountName)"
```

Often it looks like:

```text
PROJECT_NUMBER-compute@developer.gserviceaccount.com
```

Open your Google Drive folder → Share → add this email.

Recommended permission:

```text
Editor
```

(Needed if TLBrain later creates technical metadata files.)

---

## 7. Configure `.env`

Create `.env` in project root:

```env
PROJECT_ID=tlbrain-prod
REGION=europe-west1

MCP_SERVICE_NAME=tlbrain-mcp
SYNC_SERVICE_NAME=tlbrain-sync

ROOT_FOLDER_URL=https://drive.google.com/drive/folders/YOUR_FOLDER_ID
```

---

## 8. Redeploy After `.env` Changes

Run again:

```bash
bash infra/deploy/deploy.sh
```

---

## 9. Find URLs

Open:

https://console.cloud.google.com/run

You should see:

```text
tlbrain-mcp
tlbrain-sync
```

---

## MCP Endpoint

```text
https://YOUR-MCP-URL.run.app/mcp
```

Use this URL in Claude / Cowork MCP settings.

---

## Sync Endpoint

Manual sync trigger:

```text
POST https://YOUR-SYNC-URL.run.app/sync
```

Health check:

```text
GET https://YOUR-SYNC-URL.run.app/
```

---

# Current Status

Implemented:

* monorepo architecture
* dual Cloud Run deployment
* MCP remote server
* sync service foundation
* Google Drive connectivity

---

# Roadmap

Planned next:

* `.docx` transcript parsing
* hashing / change detection
* vector indexing
* semantic retrieval improvements
* production scheduler

---

# Vision

TLBrain aims to become a personal memory layer for Claude — focused, cheap, private, and practical.


