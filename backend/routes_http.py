# -*- coding: utf-8 -*-
"""All plain HTTP (non-websocket) routes."""
import time
import traceback
from datetime import datetime

from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse, JSONResponse

from . import config, database
from .ws_manager import debug_log, broadcast_to_clients, thinking_snapshot, is_thinking, clear_thinking

router = APIRouter()


@router.get("/")
async def get_dashboard():
    try:
        with open(config.INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
    except Exception as e:
        html_content = (
            "<h3>index.html not found at path: "
            f"{config.INDEX_HTML_PATH} ({e})</h3>"
        )

    config_injection = f"""
    <script>
        window.JARVIS_MODE = "{config.JARVIS_MODE}";
    </script>
    </head>
    """
    html_content = html_content.replace("</head>", config_injection)

    return HTMLResponse(
        html_content,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/api/config")
async def get_config():
    return JSONResponse({"status": "success", "mode": config.JARVIS_MODE})


@router.get("/api/sessions")
async def get_sessions():
    try:
        rows = database.list_sessions()
        sessions = []
        today_str = datetime.now(config.IST).strftime("%Y-%m-%d")

        for row in rows:
            c_date = row["created_at"].split(" ")[0] if row["created_at"] else today_str
            category = "Today" if c_date == today_str else "Previous"
            sessions.append({
                "session_id": row["session_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "pinned": bool(row["pinned"]),
                "msg_count": row["msg_count"],
                "category": category,
            })
        return JSONResponse({"status": "success", "sessions": sessions})
    except Exception as e:
        err_str = traceback.format_exc()
        debug_log(f"get_sessions Error:\n{err_str}", "bold red")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.get("/api/history")
async def get_history(session_id: str = "main_session"):
    try:
        rows = database.get_history_rows(session_id)
        history = [dict(row) for row in rows]
        return JSONResponse({
            "status": "success",
            "session_id": session_id,
            "history": history,
            "is_thinking": is_thinking(session_id),
        })
    except Exception as e:
        err_str = traceback.format_exc()
        debug_log(f"get_history Error:\n{err_str}", "bold red")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.post("/api/sessions/new")
async def create_new_session():
    session_id = f"session_{int(time.time())}"
    try:
        database.create_new_session(session_id)
        return JSONResponse({
            "status": "success",
            "session_id": session_id,
            "title": "New Conversation",
        })
    except Exception as e:
        err_str = traceback.format_exc()
        debug_log(f"create_new_session Error:\n{err_str}", "bold red")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.patch("/api/sessions/{session_id}/rename")
async def rename_session(session_id: str, payload: dict):
    new_title = (payload or {}).get("title", "").strip()
    if not new_title:
        return JSONResponse(
            {"status": "error", "message": "Title cannot be empty"}, status_code=400
        )
    try:
        updated = database.rename_session(session_id, new_title)
        if updated == 0:
            return JSONResponse(
                {"status": "error", "message": "Session not found"}, status_code=404
            )
        broadcast_to_clients({
            "type": "session_renamed",
            "session_id": session_id,
            "title": new_title,
        })
        return JSONResponse({"status": "success", "session_id": session_id, "title": new_title})
    except Exception as e:
        err_str = traceback.format_exc()
        debug_log(f"rename_session Error:\n{err_str}", "bold red")
        broadcast_to_clients({
            "type": "system_error",
            "source": "rename_session",
            "error": str(e),
            "traceback": err_str,
        })
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.patch("/api/sessions/{session_id}/pin")
async def pin_session(session_id: str, payload: dict = None):
    try:
        new_pinned = database.pin_session(session_id, payload)
        if new_pinned is None:
            return JSONResponse(
                {"status": "error", "message": "Session not found"}, status_code=404
            )
        broadcast_to_clients({
            "type": "session_pinned",
            "session_id": session_id,
            "pinned": bool(new_pinned),
        })
        return JSONResponse({
            "status": "success",
            "session_id": session_id,
            "pinned": bool(new_pinned),
        })
    except Exception as e:
        err_str = traceback.format_exc()
        debug_log(f"pin_session Error:\n{err_str}", "bold red")
        broadcast_to_clients({
            "type": "system_error",
            "source": "pin_session",
            "error": str(e),
            "traceback": err_str,
        })
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id == "main_session":
        return JSONResponse(
            {"status": "error", "message": "Cannot delete the default session"},
            status_code=400,
        )
    try:
        deleted = database.delete_session(session_id)
        clear_thinking(session_id)

        if deleted == 0:
            return JSONResponse(
                {"status": "error", "message": "Session not found"}, status_code=404
            )

        broadcast_to_clients({"type": "session_deleted", "session_id": session_id})
        return JSONResponse({"status": "success", "session_id": session_id})
    except Exception as e:
        err_str = traceback.format_exc()
        debug_log(f"delete_session Error:\n{err_str}", "bold red")
        broadcast_to_clients({
            "type": "system_error",
            "source": "delete_session",
            "error": str(e),
            "traceback": err_str,
        })
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.get("/manifest.json")
async def get_manifest():
    return JSONResponse({
        "name": "JARVIS AI OS",
        "short_name": "JARVIS",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#090a0f",
        "theme_color": "#090a0f",
        "icons": [
            {
                "src": "https://cdn-icons-png.flaticon.com/512/4712/4712035.png",
                "sizes": "192x192",
                "type": "image/png",
            }
        ],
    })


@router.get("/sw.js")
async def get_sw():
    sw_script = """
    self.addEventListener('install', (e) => self.skipWaiting());
    self.addEventListener('activate', (e) => clients.claim());
    self.addEventListener('fetch', (e) => e.respondWith(fetch(e.request)));
    """
    return Response(content=sw_script, media_type="application/javascript")
