from __future__ import annotations

import threading
import time
import uuid

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Episode:
    """
    A single remembered experience.

    An episode describes something that happened to JARVIS.
    """

    episode_id: str
    timestamp: float

    event_type: str

    context: Dict[str, Any]
    action: Optional[Dict[str, Any]]
    outcome: Optional[Dict[str, Any]]

    importance: float = 0.5
    confidence: float = 1.0

    source: Optional[str] = None

    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EpisodicMemory:
    """
    Short/medium-term experience memory of JARVIS.

    Responsibilities:
        - Store experiences.
        - Retrieve recent experiences.
        - Retrieve by event type.
        - Retrieve by tags.
        - Mark important experiences.
        - Provide serializable snapshots.

    This class does NOT decide what an experience means.

    Interpretation belongs to higher-level memory/learning organs.
    """

    VERSION = "0.1.0"

    def __init__(
        self,
        max_episodes: int = 5000,
    ):
        self.max_episodes = max(
            100,
            int(max_episodes),
        )

        self._episodes: List[Episode] = []

        self._lock = threading.RLock()

        self.created_at = time.time()
        self.updated_at = self.created_at

    # =============================================================
    # STORE
    # =============================================================

    def remember(
        self,
        event_type: str,
        context: Optional[Dict[str, Any]] = None,
        action: Optional[Dict[str, Any]] = None,
        outcome: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        confidence: float = 1.0,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        timestamp: Optional[float] = None,
    ) -> Episode:
        """
        Create and store an episode.
        """

        if not event_type:
            raise ValueError(
                "event_type cannot be empty."
            )

        episode = Episode(
            episode_id=str(uuid.uuid4()),
            timestamp=(
                time.time()
                if timestamp is None
                else float(timestamp)
            ),
            event_type=str(event_type).upper().strip(),
            context=dict(context or {}),
            action=dict(action) if action else None,
            outcome=dict(outcome) if outcome else None,
            importance=self._clamp(importance),
            confidence=self._clamp(confidence),
            source=source,
            tags=list(tags or []),
        )

        with self._lock:

            self._episodes.append(episode)

            if len(self._episodes) > self.max_episodes:

                # Remove least important old memory first.
                self._prune()

            self.updated_at = time.time()

        return episode

    # =============================================================
    # RECENT
    # =============================================================

    def recent(
        self,
        limit: int = 20,
    ) -> List[Episode]:
        """
        Return most recent episodes.
        """

        if limit <= 0:
            return []

        with self._lock:

            return list(
                self._episodes[-limit:]
            )

    # =============================================================
    # EVENT TYPE
    # =============================================================

    def find_by_event(
        self,
        event_type: str,
        limit: int = 20,
    ) -> List[Episode]:

        event_type = str(
            event_type
        ).upper().strip()

        with self._lock:

            results = [
                episode
                for episode in self._episodes
                if episode.event_type == event_type
            ]

        return results[-limit:]

    # =============================================================
    # TAG SEARCH
    # =============================================================

    def find_by_tag(
        self,
        tag: str,
        limit: int = 20,
    ) -> List[Episode]:

        tag = str(tag).lower().strip()

        with self._lock:

            results = [
                episode
                for episode in self._episodes
                if tag in [
                    item.lower()
                    for item in episode.tags
                ]
            ]

        return results[-limit:]

    # =============================================================
    # IMPORTANT MEMORY
    # =============================================================

    def important(
        self,
        threshold: float = 0.7,
        limit: int = 20,
    ) -> List[Episode]:

        threshold = self._clamp(threshold)

        with self._lock:

            results = [
                episode
                for episode in self._episodes
                if episode.importance >= threshold
            ]

        results.sort(
            key=lambda episode: (
                episode.importance,
                episode.timestamp,
            ),
            reverse=True,
        )

        return results[:limit]

    # =============================================================
    # GET BY ID
    # =============================================================

    def get(
        self,
        episode_id: str,
    ) -> Optional[Episode]:

        with self._lock:

            for episode in self._episodes:

                if episode.episode_id == episode_id:
                    return episode

        return None

    # =============================================================
    # UPDATE IMPORTANCE
    # =============================================================

    def update_importance(
        self,
        episode_id: str,
        importance: float,
    ) -> bool:

        with self._lock:

            episode = self.get(
                episode_id
            )

            if episode is None:
                return False

            episode.importance = self._clamp(
                importance
            )

            self.updated_at = time.time()

            return True

    # =============================================================
    # UPDATE CONFIDENCE
    # =============================================================

    def update_confidence(
        self,
        episode_id: str,
        confidence: float,
    ) -> bool:

        with self._lock:

            episode = self.get(
                episode_id
            )

            if episode is None:
                return False

            episode.confidence = self._clamp(
                confidence
            )

            self.updated_at = time.time()

            return True

    # =============================================================
    # COUNT
    # =============================================================

    @property
    def count(self) -> int:

        with self._lock:
            return len(self._episodes)

    # =============================================================
    # CLEAR
    # =============================================================

    def clear(self) -> None:

        with self._lock:

            self._episodes.clear()
            self.updated_at = time.time()

    # =============================================================
    # PRUNE
    # =============================================================

    def _prune(self) -> None:
        """
        Keep memory bounded.

        Recent + important experiences survive longer.
        """

        if len(self._episodes) <= self.max_episodes:
            return

        scored = sorted(
            self._episodes,
            key=lambda episode: (
                episode.importance,
                episode.timestamp,
            ),
            reverse=True,
        )

        self._episodes = scored[
            :self.max_episodes
        ]

        self._episodes.sort(
            key=lambda episode: episode.timestamp
        )

    # =============================================================
    # SNAPSHOT
    # =============================================================

    def snapshot(
        self,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:

        with self._lock:

            episodes = self._episodes

            if limit is not None:
                episodes = episodes[-max(0, limit):]

            return {
                "version": self.VERSION,
                "count": len(self._episodes),
                "max_episodes": self.max_episodes,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "episodes": [
                    episode.to_dict()
                    for episode in episodes
                ],
            }

    # =============================================================
    # RESTORE
    # =============================================================

    def restore(
        self,
        snapshot: Dict[str, Any],
    ) -> None:

        if not isinstance(snapshot, dict):
            return

        raw_episodes = snapshot.get(
            "episodes",
            [],
        )

        restored = []

        for data in raw_episodes:

            if not isinstance(data, dict):
                continue

            try:

                episode = Episode(
                    episode_id=data["episode_id"],
                    timestamp=float(
                        data["timestamp"]
                    ),
                    event_type=data["event_type"],
                    context=dict(
                        data.get("context", {})
                    ),
                    action=data.get("action"),
                    outcome=data.get("outcome"),
                    importance=self._clamp(
                        data.get("importance", 0.5)
                    ),
                    confidence=self._clamp(
                        data.get("confidence", 1.0)
                    ),
                    source=data.get("source"),
                    tags=list(
                        data.get("tags", [])
                    ),
                )

                restored.append(episode)

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

        with self._lock:

            self._episodes = restored[
                -self.max_episodes:
            ]

            self.updated_at = time.time()

    # =============================================================
    # HELPERS
    # =============================================================

    @staticmethod
    def _clamp(value: float) -> float:

        try:
            value = float(value)

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        return max(
            0.0,
            min(1.0, value),
        )