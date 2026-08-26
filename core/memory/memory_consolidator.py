from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class MemoryConsolidator:
    """
    Converts important/repeated episodic experiences into
    semantic knowledge.

    This is NOT the learning engine.

    Responsibilities:
        - inspect episodic memories
        - identify consolidation candidates
        - create semantic knowledge
        - avoid uncontrolled duplication
        - emit consolidation events

    Later:
        ExperienceEngine / SelfEvaluator can provide
        better learning signals.
    """

    VERSION = "0.1.0"

    def __init__(
        self,
        memory_manager,
        event_bus=None,
        importance_threshold: float = 0.70,
        confidence_threshold: float = 0.60,
        min_repetitions: int = 2,
    ):
        self.memory = memory_manager
        self.events = event_bus

        self.importance_threshold = importance_threshold
        self.confidence_threshold = confidence_threshold
        self.min_repetitions = min_repetitions

        self.last_run_at: Optional[float] = None
        self.last_result: Optional[Dict[str, Any]] = None
        self.run_count = 0

    # =============================================================
    # CONSOLIDATE
    # =============================================================

    def consolidate(
        self,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Inspect important episodic memories and attempt
        to convert useful patterns into semantic knowledge.
        """

        started_at = time.time()

        episodes = self.memory.important_experiences(
            threshold=self.importance_threshold,
            limit=limit,
        )

        candidates = []

        for episode in episodes:

            if self._is_candidate(episode):
                candidates.append(episode)

        consolidated = []

        for episode in candidates:

            knowledge = self._consolidate_episode(
                episode
            )

            if knowledge is not None:
                consolidated.append(
                    {
                        "episode_id": episode.episode_id,
                        "knowledge_id": knowledge.knowledge_id,
                    }
                )

        self.run_count += 1
        self.last_run_at = time.time()

        result = {
            "success": True,
            "examined": len(episodes),
            "candidates": len(candidates),
            "consolidated": len(consolidated),
            "items": consolidated,
            "duration": time.time() - started_at,
            "timestamp": self.last_run_at,
        }

        self.last_result = result

        self._emit(
            "MEMORY_CONSOLIDATION_COMPLETED",
            result,
        )

        return result

    # =============================================================
    # CANDIDATE CHECK
    # =============================================================

    def _is_candidate(
        self,
        episode,
    ) -> bool:
        """
        Decide whether an episode is useful enough
        for semantic consolidation.
        """

        importance = getattr(
            episode,
            "importance",
            0.0,
        )

        confidence = getattr(
            episode,
            "confidence",
            0.0,
        )

        if importance < self.importance_threshold:
            return False

        if confidence < self.confidence_threshold:
            return False

        return True

    # =============================================================
    # EPISODE → KNOWLEDGE
    # =============================================================

    def _consolidate_episode(
        self,
        episode,
    ):
        """
        Convert one episode into semantic knowledge.

        Current implementation uses structured episode data.
        A future KnowledgeBuilder can replace this logic.
        """

        context = getattr(
            episode,
            "context",
            None,
        ) or {}

        action = getattr(
            episode,
            "action",
            None,
        ) or {}

        outcome = getattr(
            episode,
            "outcome",
            None,
        ) or {}

        event_type = getattr(
            episode,
            "event_type",
            "UNKNOWN",
        )

        # ---------------------------------------------------------
        # Need meaningful information
        # ---------------------------------------------------------

        if not context and not action and not outcome:
            return None

        subject = self._extract_subject(
            context=context,
            action=action,
            outcome=outcome,
        )

        if not subject:
            return None

        predicate = (
            f"experienced_{event_type.lower()}"
        )

        value = {
            "context": context,
            "action": action,
            "outcome": outcome,
        }

        confidence = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        episode,
                        "confidence",
                        0.5,
                    )
                ),
            ),
        )

        importance = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        episode,
                        "importance",
                        0.5,
                    )
                ),
            ),
        )

        tags = list(
            getattr(
                episode,
                "tags",
                [],
            ) or []
        )

        # ---------------------------------------------------------
        # Store semantic knowledge
        # ---------------------------------------------------------

        knowledge = self.memory.remember_knowledge(
            subject=subject,
            predicate=predicate,
            value=value,
            confidence=confidence,
            importance=importance,
            source="memory_consolidator",
            tags=tags,
        )

        self._emit(
            "MEMORY_CONSOLIDATED",
            {
                "episode_id": episode.episode_id,
                "knowledge_id": knowledge.knowledge_id,
                "subject": subject,
                "predicate": predicate,
            },
        )

        return knowledge

    # =============================================================
    # SUBJECT EXTRACTION
    # =============================================================

    @staticmethod
    def _extract_subject(
        context: Dict[str, Any],
        action: Dict[str, Any],
        outcome: Dict[str, Any],
    ) -> Optional[str]:
        """
        Extract a stable subject from structured experience data.

        Prefer explicit subject/entity fields.
        """

        for container in (
            context,
            action,
            outcome,
        ):

            if not isinstance(
                container,
                dict,
            ):
                continue

            for key in (
                "subject",
                "entity",
                "topic",
                "name",
            ):

                value = container.get(key)

                if value is not None:

                    value = str(value).strip()

                    if value:
                        return value

        return None

    # =============================================================
    # CONSOLIDATE SINGLE EPISODE
    # =============================================================

    def consolidate_episode(
        self,
        episode_id: str,
    ) -> Optional[Any]:
        """
        Manually consolidate one known episode.
        """

        episodes = self.memory.find_experiences(
            limit=1000,
        )

        for episode in episodes:

            if episode.episode_id == episode_id:

                if not self._is_candidate(
                    episode
                ):
                    return None

                return self._consolidate_episode(
                    episode
                )

        return None

    # =============================================================
    # STATUS
    # =============================================================

    def status(self) -> Dict[str, Any]:

        return {
            "version": self.VERSION,
            "run_count": self.run_count,
            "last_run_at": self.last_run_at,
            "last_result": self.last_result,
            "importance_threshold": (
                self.importance_threshold
            ),
            "confidence_threshold": (
                self.confidence_threshold
            ),
            "min_repetitions": (
                self.min_repetitions
            ),
        }

    # =============================================================
    # EVENT BUS
    # =============================================================

    def _emit(
        self,
        event_name: str,
        payload: Any = None,
    ) -> None:

        if self.events is None:
            return

        safe_emit = getattr(
            self.events,
            "safe_emit",
            None,
        )

        if callable(safe_emit):

            safe_emit(
                event_name,
                payload,
                source="memory_consolidator",
            )