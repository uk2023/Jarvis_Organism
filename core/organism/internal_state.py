from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class InternalState:
    """
    Runtime state of the JARVIS organism.

    This is NOT personality and NOT memory.

    Memory answers:
        "What happened before?"

    InternalState answers:
        "What is happening inside the organism RIGHT NOW?"

    The state is intentionally model-independent.
    """

    VERSION = "0.1.0"

    def __init__(self):
        # ---------------------------------------------------------
        # Lifecycle
        # ---------------------------------------------------------
        self.mode = "BOOT"
        self.running = False

        # ---------------------------------------------------------
        # Attention / cognition
        # ---------------------------------------------------------
        self.attention: Optional[str] = None
        self.focus: Optional[str] = None
        self.cognitive_load = 0.0
        self.uncertainty = 0.0
        self.confidence = 0.0

        # ---------------------------------------------------------
        # Current activity
        # ---------------------------------------------------------
        self.active_task: Optional[Dict[str, Any]] = None
        self.current_goal: Optional[Dict[str, Any]] = None
        self.pending_tasks: List[Dict[str, Any]] = []

        # ---------------------------------------------------------
        # Learning
        # ---------------------------------------------------------
        self.learning_state = "IDLE"
        self.last_learning_result: Optional[Dict[str, Any]] = None
        self.learning_count = 0

        # ---------------------------------------------------------
        # Conversation
        # ---------------------------------------------------------
        self.conversation_active = False
        self.last_user_input: Optional[str] = None
        self.last_response: Optional[str] = None

        # ---------------------------------------------------------
        # Perception / routing
        # ---------------------------------------------------------
        self.last_intent: Optional[str] = None
        self.last_route: Optional[str] = None
        self.last_perception: Optional[Dict[str, Any]] = None

        # ---------------------------------------------------------
        # Organism events
        # ---------------------------------------------------------
        self.last_event: Optional[Dict[str, Any]] = None
        self.event_count = 0

        # ---------------------------------------------------------
        # Autonomy
        # ---------------------------------------------------------
        self.autonomy_enabled = False
        self.autonomy_state = "DORMANT"
        self.last_autonomous_action: Optional[Dict[str, Any]] = None

        # ---------------------------------------------------------
        # Runtime timestamps
        # ---------------------------------------------------------
        now = time.time()

        self.created_at = now
        self.updated_at = now
        self.last_activity_at = now

    # =============================================================
    # GENERIC UPDATE
    # =============================================================

    def update(self, **changes) -> None:
        """
        Update known state fields.

        Unknown fields are intentionally rejected rather than silently
        creating random state variables.
        """

        for key, value in changes.items():

            if not hasattr(self, key):
                raise AttributeError(
                    f"InternalState has no field '{key}'"
                )

            setattr(self, key, value)

        now = time.time()

        self.updated_at = now
        self.last_activity_at = now

    # =============================================================
    # EVENT
    # =============================================================

    def record_event(
        self,
        event_name: str,
        payload: Any = None,
        source: Optional[str] = None,
    ) -> None:
        """
        Record the latest organism event.
        """

        self.last_event = {
            "name": event_name,
            "payload": payload,
            "source": source,
            "timestamp": time.time(),
        }

        self.event_count += 1

        self.updated_at = time.time()
        self.last_activity_at = self.updated_at

    # =============================================================
    # USER INTERACTION
    # =============================================================

    def record_user_input(
        self,
        text: str,
        intent: Optional[str] = None,
        route: Optional[str] = None,
    ) -> None:

        self.last_user_input = text
        self.last_intent = intent
        self.last_route = route
        self.conversation_active = True

        self.updated_at = time.time()
        self.last_activity_at = self.updated_at

    def record_response(self, response: str) -> None:

        self.last_response = response

        self.updated_at = time.time()
        self.last_activity_at = self.updated_at

    # =============================================================
    # COGNITIVE STATE
    # =============================================================

    def set_attention(
        self,
        attention: Optional[str],
        confidence: float = 0.0,
    ) -> None:

        self.attention = attention
        self.confidence = self._clamp(confidence)

        self.updated_at = time.time()

    def set_uncertainty(self, value: float) -> None:
        self.uncertainty = self._clamp(value)

        self.updated_at = time.time()

    def set_cognitive_load(self, value: float) -> None:
        self.cognitive_load = self._clamp(value)

        self.updated_at = time.time()

    # =============================================================
    # TASK STATE
    # =============================================================

    def start_task(
        self,
        task_id: str,
        description: str,
        source: str = "USER",
    ) -> None:

        self.active_task = {
            "id": task_id,
            "description": description,
            "source": source,
            "started_at": time.time(),
            "status": "RUNNING",
        }

        self.updated_at = time.time()

    def finish_task(
        self,
        success: bool,
        result: Any = None,
    ) -> None:

        if self.active_task is None:
            return

        self.active_task["status"] = (
            "COMPLETED" if success else "FAILED"
        )

        self.active_task["success"] = success
        self.active_task["result"] = result
        self.active_task["finished_at"] = time.time()

        self.updated_at = time.time()

    def clear_active_task(self) -> None:
        self.active_task = None
        self.updated_at = time.time()

    # =============================================================
    # GOALS
    # =============================================================

    def set_goal(
        self,
        goal_id: str,
        description: str,
        source: str = "AUTONOMOUS",
    ) -> None:

        self.current_goal = {
            "id": goal_id,
            "description": description,
            "source": source,
            "status": "ACTIVE",
            "created_at": time.time(),
        }

        self.updated_at = time.time()

    def clear_goal(self) -> None:
        self.current_goal = None
        self.updated_at = time.time()

    # =============================================================
    # LEARNING
    # =============================================================

    def begin_learning(self) -> None:
        self.learning_state = "LEARNING"
        self.updated_at = time.time()

    def record_learning(
        self,
        result: Dict[str, Any],
    ) -> None:

        self.learning_state = "UPDATED"
        self.last_learning_result = result
        self.learning_count += 1

        self.updated_at = time.time()

    def finish_learning(self) -> None:
        self.learning_state = "IDLE"
        self.updated_at = time.time()

    # =============================================================
    # AUTONOMY
    # =============================================================

    def enable_autonomy(self) -> None:
        self.autonomy_enabled = True
        self.autonomy_state = "READY"

        self.updated_at = time.time()

    def disable_autonomy(self) -> None:
        self.autonomy_enabled = False
        self.autonomy_state = "DORMANT"

        self.updated_at = time.time()

    def record_autonomous_action(
        self,
        action: Dict[str, Any],
    ) -> None:

        self.last_autonomous_action = action
        self.autonomy_state = "ACTIVE"

        self.updated_at = time.time()

    # =============================================================
    # PERCEPTION
    # =============================================================

    def record_perception(
        self,
        perception: Dict[str, Any],
    ) -> None:

        self.last_perception = perception

        self.updated_at = time.time()

    # =============================================================
    # SNAPSHOT
    # =============================================================

    def snapshot(self) -> Dict[str, Any]:
        """
        Return serializable state.

        This snapshot will later be persisted to SQLite so that
        JARVIS can restore important state after Android restart.
        """

        return {
            "version": self.VERSION,

            "lifecycle": {
                "mode": self.mode,
                "running": self.running,
            },

            "cognition": {
                "attention": self.attention,
                "focus": self.focus,
                "cognitive_load": self.cognitive_load,
                "uncertainty": self.uncertainty,
                "confidence": self.confidence,
            },

            "activity": {
                "active_task": self.active_task,
                "current_goal": self.current_goal,
                "pending_tasks": list(self.pending_tasks),
            },

            "learning": {
                "state": self.learning_state,
                "last_result": self.last_learning_result,
                "count": self.learning_count,
            },

            "conversation": {
                "active": self.conversation_active,
                "last_user_input": self.last_user_input,
                "last_response": self.last_response,
            },

            "perception": {
                "last_intent": self.last_intent,
                "last_route": self.last_route,
                "last_perception": self.last_perception,
            },

            "events": {
                "last_event": self.last_event,
                "count": self.event_count,
            },

            "autonomy": {
                "enabled": self.autonomy_enabled,
                "state": self.autonomy_state,
                "last_action": self.last_autonomous_action,
            },

            "timestamps": {
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "last_activity_at": self.last_activity_at,
            },
        }

    # =============================================================
    # RESTORE
    # =============================================================

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """
        Restore state from a previously persisted snapshot.

        This is intentionally conservative: only known fields are
        restored.
        """

        if not isinstance(snapshot, dict):
            return

        lifecycle = snapshot.get("lifecycle", {})
        cognition = snapshot.get("cognition", {})
        activity = snapshot.get("activity", {})
        learning = snapshot.get("learning", {})
        conversation = snapshot.get("conversation", {})
        perception = snapshot.get("perception", {})
        events = snapshot.get("events", {})
        autonomy = snapshot.get("autonomy", {})
        timestamps = snapshot.get("timestamps", {})

        self._restore_fields(
            lifecycle,
            ["mode", "running"],
        )

        self._restore_fields(
            cognition,
            [
                "attention",
                "focus",
                "cognitive_load",
                "uncertainty",
                "confidence",
            ],
        )

        self._restore_fields(
            activity,
            [
                "active_task",
                "current_goal",
                "pending_tasks",
            ],
        )

        self._restore_fields(
            learning,
            [
                "learning_state",
                "last_learning_result",
                "learning_count",
            ],
        )

        self._restore_fields(
            conversation,
            [
                "conversation_active",
                "last_user_input",
                "last_response",
            ],
        )

        self._restore_fields(
            perception,
            [
                "last_intent",
                "last_route",
                "last_perception",
            ],
        )

        self._restore_fields(
            events,
            [
                "last_event",
                "event_count",
            ],
        )

        self._restore_fields(
            autonomy,
            [
                "autonomy_enabled",
                "autonomy_state",
                "last_autonomous_action",
            ],
        )

        self._restore_fields(
            timestamps,
            [
                "created_at",
                "updated_at",
                "last_activity_at",
            ],
        )

        self.updated_at = time.time()

    # =============================================================
    # INTERNAL HELPERS
    # =============================================================

    def _restore_fields(
        self,
        source: Dict[str, Any],
        fields: List[str],
    ) -> None:

        for field in fields:
            if field in source and hasattr(self, field):
                setattr(self, field, source[field])

    @staticmethod
    def _clamp(value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(0.0, min(1.0, value))