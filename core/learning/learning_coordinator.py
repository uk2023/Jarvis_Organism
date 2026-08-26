from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class LearningCoordinator:
    """
    Central learning orchestration organ of JARVIS.

    Responsibilities:

        Experience
             ↓
        SelfEvaluator
             ↓
        KnowledgeBuilder
             ↓
        Candidate Knowledge
             ↓
        Explicit Acceptance
             ↓
        Semantic Memory

    Periodically:

        Episodic Memory
             ↓
        MemoryConsolidator
             ↓
        Semantic Memory

    Important architecture rules:

        - ExperienceEngine records experiences.
        - SelfEvaluator evaluates experiences.
        - KnowledgeBuilder creates knowledge candidates.
        - LearningCoordinator orchestrates these organs.
        - MemoryConsolidator performs separate long-term consolidation.
        - Knowledge is NOT automatically trusted.
        - auto_accept is disabled by default.
        - No source-code modification happens here.
        - No autonomous action execution happens here.
    """

    VERSION = "0.2.0"

    def __init__(
        self,
        evaluator=None,
        knowledge_builder=None,
        consolidator=None,
        memory_manager=None,
        event_bus=None,
        internal_state=None,
    ):
        self.evaluator = evaluator
        self.knowledge_builder = knowledge_builder
        self.consolidator = consolidator
        self.memory = memory_manager

        self.events = event_bus
        self.state = internal_state

        # ---------------------------------------------------------
        # Runtime statistics
        # ---------------------------------------------------------

        self.learning_count = 0
        self.evaluation_count = 0
        self.knowledge_build_count = 0
        self.knowledge_accept_count = 0
        self.knowledge_reject_count = 0
        self.consolidation_count = 0

        self.last_learning_at: Optional[float] = None
        self.last_result: Optional[Dict[str, Any]] = None

        self.running = True

    # =============================================================
    # LEARN
    # =============================================================

    def learn(
        self,
        experience: Dict[str, Any],
        auto_accept: bool = False,
    ) -> Dict[str, Any]:
        """
        Run one controlled learning cycle.

        Pipeline:

            experience
                ↓
            SelfEvaluator
                ↓
            KnowledgeBuilder
                ↓
            candidate
                ↓
            optional explicit acceptance

        Memory consolidation is NOT automatically performed here.
        """

        if not self.running:
            raise RuntimeError(
                "LearningCoordinator is stopped."
            )

        if not isinstance(experience, dict):
            raise TypeError(
                "experience must be a dictionary"
            )

        started_at = time.time()

        result: Dict[str, Any] = {
            "type": "LEARNING_CYCLE",
            "success": False,
            "experience": experience,
            "evaluation": None,
            "knowledge": None,
            "accepted": False,
            "duration": 0.0,
            "timestamp": None,
        }

        # ---------------------------------------------------------
        # 1. SELF EVALUATION
        # ---------------------------------------------------------

        if self.evaluator is None:
            raise RuntimeError(
                "SelfEvaluator is not connected."
            )

        evaluation = self.evaluator.evaluate(
            experience
        )

        result["evaluation"] = evaluation

        self.evaluation_count += 1

        # ---------------------------------------------------------
        # 2. KNOWLEDGE BUILDING
        # ---------------------------------------------------------

        if self.knowledge_builder is not None:

            candidate = self.knowledge_builder.build(
                experience=experience,
                evaluation=evaluation,
            )

            result["knowledge"] = candidate

            if candidate is not None:

                self.knowledge_build_count += 1

                # -------------------------------------------------
                # 3. OPTIONAL ACCEPTANCE
                # -------------------------------------------------

                if auto_accept:

                    accepted = self.accept_knowledge(
                        candidate["id"]
                    )

                    result["accepted"] = (
                        accepted.get("status")
                        == "ACCEPTED"
                    )

        # ---------------------------------------------------------
        # 4. COMPLETE
        # ---------------------------------------------------------

        self.learning_count += 1

        self.last_learning_at = time.time()

        result["success"] = True
        result["duration"] = (
            time.time() - started_at
        )
        result["timestamp"] = (
            self.last_learning_at
        )

        self.last_result = result

        self._emit(
            "LEARNING_COMPLETED",
            result,
        )

        return result

    # =============================================================
    # EVALUATE ONLY
    # =============================================================

    def evaluate(
        self,
        experience: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run only SelfEvaluator.
        """

        if not self.running:
            raise RuntimeError(
                "LearningCoordinator is stopped."
            )

        if self.evaluator is None:
            raise RuntimeError(
                "SelfEvaluator is not connected."
            )

        result = self.evaluator.evaluate(
            experience
        )

        self.evaluation_count += 1

        self._emit(
            "LEARNING_EVALUATION_COMPLETED",
            result,
        )

        return result

    # =============================================================
    # BUILD KNOWLEDGE
    # =============================================================

    def build_knowledge(
        self,
        experience: Dict[str, Any],
        evaluation: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Build a candidate without accepting it.
        """

        if self.knowledge_builder is None:
            raise RuntimeError(
                "KnowledgeBuilder is not connected."
            )

        if evaluation is None:
            evaluation = self.evaluate(
                experience
            )

        candidate = self.knowledge_builder.build(
            experience=experience,
            evaluation=evaluation,
        )

        if candidate is not None:

            self.knowledge_build_count += 1

            self._emit(
                "LEARNING_KNOWLEDGE_BUILT",
                candidate,
            )

        return candidate

    # =============================================================
    # ACCEPT KNOWLEDGE
    # =============================================================

    def accept_knowledge(
        self,
        knowledge_id: str,
    ) -> Dict[str, Any]:
        """
        Explicitly accept a KnowledgeBuilder candidate.

        KnowledgeBuilder owns the actual MemoryManager handoff.
        """

        if self.knowledge_builder is None:
            raise RuntimeError(
                "KnowledgeBuilder is not connected."
            )

        candidate = self.knowledge_builder.accept(
            knowledge_id
        )

        if candidate.get("status") == "ACCEPTED":

            self.knowledge_accept_count += 1

            self._emit(
                "LEARNING_KNOWLEDGE_ACCEPTED",
                candidate,
            )

        return candidate

    # =============================================================
    # REJECT KNOWLEDGE
    # =============================================================

    def reject_knowledge(
        self,
        knowledge_id: str,
        reason: str = "",
    ) -> Dict[str, Any]:
        """
        Explicitly reject a candidate.
        """

        if self.knowledge_builder is None:
            raise RuntimeError(
                "KnowledgeBuilder is not connected."
            )

        candidate = self.knowledge_builder.reject(
            knowledge_id=knowledge_id,
            reason=reason,
        )

        self.knowledge_reject_count += 1

        self._emit(
            "LEARNING_KNOWLEDGE_REJECTED",
            candidate,
        )

        return candidate

    # =============================================================
    # MEMORY CONSOLIDATION
    # =============================================================

    def consolidate(
        self,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Run episodic → semantic memory consolidation.

        This is intentionally separate from learn().
        """

        if self.consolidator is None:
            raise RuntimeError(
                "MemoryConsolidator is not connected."
            )

        result = self.consolidator.consolidate(
            limit=limit
        )

        self.consolidation_count += 1

        self._emit(
            "LEARNING_CONSOLIDATION_COMPLETED",
            result,
        )

        return result

    # =============================================================
    # LEARN + CONSOLIDATE
    # =============================================================

    def learn_and_consolidate(
        self,
        experience: Dict[str, Any],
        auto_accept: bool = False,
        consolidation_limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Controlled testing/helper pipeline.

        Normal runtime should generally call learn()
        and consolidate() independently.
        """

        learning = self.learn(
            experience=experience,
            auto_accept=auto_accept,
        )

        consolidation = self.consolidate(
            limit=consolidation_limit
        )

        return {
            "learning": learning,
            "consolidation": consolidation,
            "timestamp": time.time(),
        }

    # =============================================================
    # CANDIDATE LOOKUP
    # =============================================================

    def get_candidate(
        self,
        knowledge_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a candidate from KnowledgeBuilder.
        """

        if self.knowledge_builder is None:
            return None

        getter = getattr(
            self.knowledge_builder,
            "get",
            None,
        )

        if not callable(getter):
            return None

        return getter(
            knowledge_id
        )

    # =============================================================
    # CANDIDATE LIST
    # =============================================================

    def list_candidates(
        self,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        List candidates managed by KnowledgeBuilder.
        """

        if self.knowledge_builder is None:
            return []

        method = getattr(
            self.knowledge_builder,
            "list_knowledge",
            None,
        )

        if not callable(method):
            return []

        return method(
            status=status,
            limit=limit,
        )

    # =============================================================
    # STATUS
    # =============================================================

    def status(self) -> Dict[str, Any]:
        """
        Return learning subsystem status.
        """

        evaluator_status = {}
        builder_status = {}
        consolidator_status = {}

        # ---------------------------------------------------------
        # Evaluator
        # ---------------------------------------------------------

        if self.evaluator is not None:

            method = getattr(
                self.evaluator,
                "statistics",
                None,
            )

            if callable(method):

                try:
                    evaluator_status = method()

                except Exception as exc:
                    evaluator_status = {
                        "error": str(exc)
                    }

        # ---------------------------------------------------------
        # KnowledgeBuilder
        # ---------------------------------------------------------

        if self.knowledge_builder is not None:

            method = getattr(
                self.knowledge_builder,
                "statistics",
                None,
            )

            if callable(method):

                try:
                    builder_status = method()

                except Exception as exc:
                    builder_status = {
                        "error": str(exc)
                    }

        # ---------------------------------------------------------
        # Consolidator
        # ---------------------------------------------------------

        if self.consolidator is not None:

            method = getattr(
                self.consolidator,
                "status",
                None,
            )

            if callable(method):

                try:
                    consolidator_status = method()

                except Exception as exc:
                    consolidator_status = {
                        "error": str(exc)
                    }

        return {
            "version": self.VERSION,
            "running": self.running,

            "learning_count": (
                self.learning_count
            ),

            "evaluation_count": (
                self.evaluation_count
            ),

            "knowledge_build_count": (
                self.knowledge_build_count
            ),

            "knowledge_accept_count": (
                self.knowledge_accept_count
            ),

            "knowledge_reject_count": (
                self.knowledge_reject_count
            ),

            "consolidation_count": (
                self.consolidation_count
            ),

            "last_learning_at": (
                self.last_learning_at
            ),

            "evaluator": evaluator_status,

            "knowledge_builder": builder_status,

            "consolidator": consolidator_status,
        }

    # =============================================================
    # RESET
    # =============================================================

    def reset_statistics(self) -> None:
        """
        Reset coordinator statistics only.

        Memory and knowledge are untouched.
        """

        self.learning_count = 0
        self.evaluation_count = 0
        self.knowledge_build_count = 0
        self.knowledge_accept_count = 0
        self.knowledge_reject_count = 0
        self.consolidation_count = 0

        self.last_learning_at = None
        self.last_result = None

    # =============================================================
    # START / STOP
    # =============================================================

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

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
                source="learning_coordinator",
            )