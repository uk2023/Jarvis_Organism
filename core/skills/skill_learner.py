from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, List


class SkillLearner:
    """
    Turns repeated, verified successful experiences into reusable
    skill *proposals*.

    It never registers a skill directly — SkillRegistry stays the
    single source of truth, and a human (or a future self-evaluator
    with a higher trust threshold) decides whether to accept a
    proposal. This keeps skill acquisition auditable instead of
    letting the organism silently rewrite its own capability set.

    An experience is expected to look like:
        {
            "action": str,
            "detail": str,
            "status": "success" | "error",
            "goal": str,
        }
    """

    def __init__(self, min_repetitions: int = 3, min_success_rate: float = 0.8):
        self.min_repetitions = min_repetitions
        self.min_success_rate = min_success_rate

    def propose(self, experiences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not experiences:
            return []

        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for exp in experiences:
            action = exp.get("action")

            if not action:
                continue

            grouped.setdefault(action, []).append(exp)

        proposals = []

        for action, group in grouped.items():
            if len(group) < self.min_repetitions:
                continue

            successes = [e for e in group if e.get("status") == "success"]
            success_rate = len(successes) / len(group)

            if success_rate < self.min_success_rate:
                continue

            proposals.append(
                {
                    "name": self._skill_name(action),
                    "based_on_action": action,
                    "repetitions": len(group),
                    "success_rate": round(success_rate, 2),
                    "example_detail": group[-1].get("detail", ""),
                    "proposed_at": time.time(),
                    "status": "proposed",
                }
            )

        return proposals

    @staticmethod
    def _skill_name(action: str) -> str:
        return "skill_" + "".join(
            c if c.isalnum() else "_" for c in action.lower()
        ).strip("_")
