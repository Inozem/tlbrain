#!/usr/bin/env python3
"""One-time setup: authorize Google Drive access and save refresh token to .env"""

import http.server
import os
import threading
import urllib.parse
import webbrowser
from pathlib import Path

import requests

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
]
REDIRECT_URI = "http://localhost:8085"
AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
ENV_FILE = Path(".env")


def _load_env() -> dict:
    if not ENV_FILE.exists():
        return {}
    env = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip()
    return env


def _set_env_var(key: str, value: str):
    content = ENV_FILE.read_text() if ENV_FILE.exists() else ""
    lines = content.splitlines()
    new_lines, found = [], False
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(new_lines) + "\n")


def main():
    env = _load_env()

    client_id = env.get("GOOGLE_CLIENT_ID") or os.environ.get("GOOGLE_CLIENT_ID", "")
    client_secret = env.get("GOOGLE_CLIENT_SECRET") or os.environ.get("GOOGLE_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        print("ERROR: GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set in .env")
        print("Get them from: Google Cloud Console → APIs & Services → OAuth 2.0 Client IDs")
        raise SystemExit(1)

    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_url = AUTH_URI + "?" + urllib.parse.urlencode(params)

    code_holder: list[str] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if "code" in qs:
                code_holder.append(qs["code"][0])
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"<h2>Authorized! You can close this tab.</h2>")
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Authorization failed.")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("localhost", 8085), _Handler)
    thread = threading.Thread(target=server.handle_request)
    thread.start()

    print("Opening browser for Google authorization...")
    webbrowser.open(auth_url)
    thread.join(timeout=120)

    if not code_holder:
        print("ERROR: No authorization code received (timeout or cancelled).")
        raise SystemExit(1)

    resp = requests.post(TOKEN_URI, data={
        "code": code_holder[0],
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    })
    resp.raise_for_status()
    tokens = resp.json()

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        print("ERROR: No refresh_token returned.")
        print("Try revoking access at myaccount.google.com/permissions and retry.")
        raise SystemExit(1)

    _set_env_var("GOOGLE_REFRESH_TOKEN", refresh_token)
    print("GOOGLE_REFRESH_TOKEN saved to .env")


if __name__ == "__main__":
    main()
