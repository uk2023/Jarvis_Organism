from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class IdleLoop:
    """
    Autonomous maintenance loop, run periodically by Heartbeat when
    the organism has no active user interaction.

    Safety model (unchanged from the original design intent):
      1. Curiosity proposes *candidates* (never executes anything).
      2. Planner turns the top candidate/goal into steps.
      3. Any step with requires_confirmation=True is NOT executed here
         — it is surfaced as a pending confirmation for the user.
      4. Steps that are safe to run are handed to the provided
         `executor` callback, and the outcome is logged as an episode.

    This class contains no model-loading and no direct file/network
    access itself — it only coordinates already-attached organs.
    """

    def __init__(
        self,
        goal_manager=None,
        curiosity=None,
        planner=None,
        scheduler=None,
        state=None,
        event_bus=None,
        store=None,
        executor=None,
        max_actions_per_step: int = 1,
    ):
        self.goal_manager = goal_manager
        self.curiosity = curiosity
        self.planner = planner
        self.scheduler = scheduler
        self.state = state
        self.events = event_bus
        self.store = store
        self.executor = executor
        self.max_actions_per_step = max_actions_per_step

        self.pending_confirmations: List[Dict[str, Any]] = []

    def step(self) -> Dict[str, Any]:
        """Run exactly one idle cycle. Called by Heartbeat/Scheduler."""

        # ---------------------------------------------------------
        # 0) Run anything the Scheduler says is due first.
        # ---------------------------------------------------------
        if self.scheduler is not None:
            for task in self.scheduler.due_tasks():
                self._run_task(task)

        # ---------------------------------------------------------
        # 1) Nothing to do if no goal_manager/curiosity attached yet.
        # ---------------------------------------------------------
        if self.goal_manager is None or self.curiosity is None or self.planner is None:
            return self._noop("autonomy organs not fully attached")

        goals = self.goal_manager.pending()

        candidates = self.curiosity.candidates(state=self.state, goals=goals)

        # Curiosity-sourced candidates become goals the first time
        # they're seen, so progress on them is tracked consistently.
        for candidate in candidates:
            self.goal_manager.add(
                text=candidate["reason"],
                priority=candidate.get("priority", 0.5),
                origin="curiosity",
            )

        target = self.goal_manager.next_goal()

        if target is None:
            return self._noop("no pending goals")

        self.goal_manager.update_status(target["id"], "active")

        steps = self.planner.plan(target)

        executed = []
        confirmations_needed = []

        for step in steps[: self.max_actions_per_step]:
            if step.get("requires_confirmation"):
                confirmations_needed.append(step)
                self.pending_confirmations.append(
                    {**step, "goal_id": target["id"], "queued_at": time.time()}
                )
                continue

            outcome = self._run_step(target, step)
            executed.append(outcome)

        if executed and not confirmations_needed:
            self.goal_manager.add_progress(
                target["id"], f"Executed: {[s['action'] for s in executed]}"
            )

        if not steps:
            self.goal_manager.update_status(target["id"], "completed")

        result = {
            "action": "IDLE_CYCLE",
            "goal": target["text"],
            "executed": executed,
            "awaiting_confirmation": confirmations_needed,
        }

        self._publish("IDLE_CYCLE_COMPLETE", result)
        return result

    # =============================================================
    # INTERNAL
    # =============================================================

    def _run_step(self, goal: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
        outcome = {"step": step, "status": "skipped", "result": None}

        if self.executor is None:
            outcome["status"] = "no_executor"
            return outcome

        try:
            result = self.executor(step)
            outcome["status"] = "success"
            outcome["result"] = result
        except Exception as exc:
            outcome["status"] = "error"
            outcome["result"] = str(exc)

        self._log_episode(goal, step, outcome)
        return outcome

    def _run_task(self, task: Dict[str, Any]) -> None:
        if self.executor is None:
            return

        try:
            self.executor(task)
        except Exception as exc:
            print(f"[IdleLoop] Scheduled task failed: {exc}")

    def _log_episode(
        self,
        goal: Dict[str, Any],
        step: Dict[str, Any],
        outcome: Dict[str, Any],
    ) -> None:
        if self.store is None:
            return

        try:
            self.store.save_episode(
                {
                    "episode_id": f"idle-{time.time()}",
                    "timestamp": time.time(),
                    "event_type": "AUTONOMOUS_STEP",
                    "context": {"goal": goal.get("text")},
                    "action": step,
                    "outcome": outcome,
                    "importance": 0.3,
                    "confidence": 1.0 if outcome["status"] == "success" else 0.4,
                    "source": "idle_loop",
                }
            )
        except Exception as exc:
            print(f"[IdleLoop] Failed to log episode: {exc}")

    def _noop(self, reason: str) -> Dict[str, Any]:
        result = {"action": "NO_OP", "reason": reason}
        self._publish("IDLE_CYCLE_NOOP", result)
        return result

    def _publish(self, name: str, payload: Any) -> None:
        if self.events is None:
            return

        emit = getattr(self.events, "emit", None)

        if callable(emit):
            try:
                emit(name, payload)
            except Exception as exc:
                print(f"[IdleLoop EventBus Error] {name}: {exc}")
