# -*- coding: utf-8 -*-
"""
Central configuration & constants for the JARVIS web server.

Everything path/constant related used to be scattered as module-level
globals at the top of the old single-file dashboard.py. Pulling it into
its own module means every other module (database, routes, server) can
import just what it needs without pulling in FastAPI app wiring.
"""
import os
from datetime import datetime, timezone, timedelta

# Root of the whole Jarvis Organism project on the device.
BASE_DIR = "/storage/emulated/0/Jarvis_Organism"

# Frontend now lives in its own folder with split html/css/js instead of a
# single index.html with everything inlined.
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
INDEX_HTML_PATH = os.path.join(FRONTEND_DIR, "index.html")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

DB_PATH = os.path.join(BASE_DIR, "database", "jarvis.db")

# System Mode: 'dev' or 'normal' (Mode 2)
JARVIS_MODE = os.environ.get("JARVIS_MODE", "dev").lower()

# Indian Standard Timezone (IST: UTC +5:30) with High Precision
IST = timezone(timedelta(hours=5, minutes=30))

# Placeholder titles that should be auto-replaced by the first real user
# message in a thread (ChatGPT/Gemini-style auto-titling).
DEFAULT_SESSION_TITLES = ("New Conversation", "New Neural Thread", "")


def get_local_ist_timestamp() -> str:
    """Returns a high-precision IST timestamp (incl. microseconds) for absolute sync."""
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S.%f")
