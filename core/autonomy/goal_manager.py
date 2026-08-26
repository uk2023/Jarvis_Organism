from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional


class GoalManager:
    """
    Tracks JARVIS's goals and persists them through the SQLiteStore's
    organism_meta table (key = 'goals'), so goals survive restarts.

    A goal is a plain dict:
        {
            "id": str,
            "text": str,
            "priority": float,   # 0..1, higher = more important
            "status": "pending" | "active" | "completed" | "abandoned",
            "origin": str,       # "user" | "curiosity" | "self"
            "created_at": float,
            "updated_at": float,
            "progress": list[str],   # short log of progress notes
        }

    No LLM calls happen here. This module only manages bookkeeping;
    the Planner decides *how* to pursue a goal.
    """

    META_KEY = "goals"

    def __init__(self, store=None):
        self.store = store
        self.goals: List[Dict[str, Any]] = []

        if self.store is not None:
            self._load()

    # =============================================================
    # PERSISTENCE
    # =============================================================

    def _load(self) -> None:
        try:
            saved = self.store.get_meta(self.META_KEY)
        except Exception:
            saved = None

        if isinstance(saved, list):
            self.goals = saved

    def _save(self) -> None:
        if self.store is None:
            return

        try:
            self.store.set_meta(self.META_KEY, self.goals)
        except Exception as exc:
            print(f"[GoalManager] Failed to persist goals: {exc}")

    # =============================================================
    # MUTATIONS
    # =============================================================

    def add(
        self,
        text: str,
        priority: float = 0.5,
        origin: str = "user",
    ) -> Dict[str, Any]:
        if not text:
            raise ValueError("Goal text cannot be empty.")

        now = time.time()

        goal = {
            "id": str(uuid.uuid4()),
            "text": text,
            "priority": max(0.0, min(1.0, priority)),
            "status": "pending",
            "origin": origin,
            "created_at": now,
            "updated_at": now,
            "progress": [],
        }

        self.goals.append(goal)
        self._save()
        return goal

    def update_status(self, goal_id: str, status: str) -> Optional[Dict[str, Any]]:
        goal = self._find(goal_id)

        if goal is None:
            return None

        goal["status"] = status
        goal["updated_at"] = time.time()
        self._save()
        return goal

    def add_progress(self, goal_id: str, note: str) -> Optional[Dict[str, Any]]:
        goal = self._find(goal_id)

        if goal is None:
            return None

        goal["progress"].append({"note": note, "timestamp": time.time()})
        goal["updated_at"] = time.time()
        self._save()
        return goal

    def remove(self, goal_id: str) -> bool:
        before = len(self.goals)
        self.goals = [g for g in self.goals if g["id"] != goal_id]

        changed = len(self.goals) != before
        if changed:
            self._save()

        return changed

    # =============================================================
    # QUERIES
    # =============================================================

    def _find(self, goal_id: str) -> Optional[Dict[str, Any]]:
        for g in self.goals:
            if g["id"] == goal_id:
                return g
        return None

    def pending(self) -> List[Dict[str, Any]]:
        return [g for g in self.goals if g.get("status") in ("pending", "active")]

    def next_goal(self) -> Optional[Dict[str, Any]]:
        """Highest-priority pending goal, oldest first as a tiebreaker."""

        candidates = self.pending()

        if not candidates:
            return None

        return sorted(
            candidates,
            key=lambda g: (-g["priority"], g["created_at"]),
        )[0]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total": len(self.goals),
            "pending": len(self.pending()),
            "completed": len([g for g in self.goals if g["status"] == "completed"]),
        }
