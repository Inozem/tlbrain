# tlbrain
> 🚧 This project is under active development.

TLBrain is a cost-efficient memory system for Claude (Cowork), designed to retrieve structured knowledge from conversation transcripts.

The system aims to enable Claude to "remember" past client conversations by integrating:
- TL;DV transcripts
- Google Drive storage
- MCP-based retrieval layer

It is designed as a lightweight, cost-efficient alternative to traditional RAG pipelines. Optimized for single-user scenarios with high volumes of client conversations.

# Quick Deploy Guide (Google Cloud)

## 1. Create Google Cloud Project

Open:

https://console.cloud.google.com/

Create a new project.

Recommended name:

tlbrain-prod

After creation copy:

PROJECT_ID

Example:

tlbrain-prod

---

## 2. Install Google Cloud CLI

Download and install:

https://docs.cloud.google.com/sdk/docs/install-sdk

After install run:

gcloud auth login

Then:

gcloud config set project YOUR_PROJECT_ID

---

## 3. Enable required services

Run:

gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com

---

## 4. Configure .env

Create `.env` in project root:

PROJECT_ID=tlbrain-prod
REGION=europe-west1
SERVICE_NAME=tlbrain-mcp

---

## 5. Deploy MCP server

Run:

bash infra/deploy.sh

---

## 6. Find MCP URL

Open:

https://console.cloud.google.com/run

Select project:

tlbrain-prod

Open service:

tlbrain-mcp

Copy URL.

Final MCP endpoint:

https://YOUR-URL.run.app/mcp
