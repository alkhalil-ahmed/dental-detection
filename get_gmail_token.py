"""
One-time helper script to obtain a Gmail API OAuth2 refresh token.

Run this ONCE on your local machine (not on Railway):
    python get_gmail_token.py

It will open your browser to authorize the app, then print the
GMAIL_REFRESH_TOKEN value to paste into Railway Variables.

Prerequisites:
    pip install google-auth-oauthlib
"""

import json
import sys

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError:
    print("Install the required package first:")
    print("  pip install google-auth-oauthlib")
    sys.exit(1)

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

def main():
    print("=" * 60)
    print("  Gmail API — OAuth2 Refresh Token Generator")
    print("=" * 60)
    print()

    client_id = input("Paste your GMAIL_CLIENT_ID: ").strip()
    client_secret = input("Paste your GMAIL_CLIENT_SECRET: ").strip()

    if not client_id or not client_secret:
        print("Error: Both client ID and secret are required.")
        sys.exit(1)

    # Build OAuth2 flow from client config dict (no file needed)
    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }

    flow = InstalledAppFlow.from_client_config(client_config, scopes=SCOPES)
    creds = flow.run_local_server(port=8090, prompt="consent", access_type="offline")

    print()
    print("=" * 60)
    print("  SUCCESS! Copy these values to Railway Variables:")
    print("=" * 60)
    print()
    print(f"  GMAIL_CLIENT_ID     = {client_id}")
    print(f"  GMAIL_CLIENT_SECRET = {client_secret}")
    print(f"  GMAIL_REFRESH_TOKEN = {creds.refresh_token}")
    print(f"  GMAIL_USER          = (the Gmail address you just authorized)")
    print(f"  EMAIL_PROVIDER      = gmail_api")
    print()

if __name__ == "__main__":
    main()
