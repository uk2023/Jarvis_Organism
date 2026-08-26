from __future__ import annotations

from typing import Optional

from .jarvis_core import JarvisCore
from .internal_state import InternalState
from .event_bus import EventBus
from .heartbeat import Heartbeat
from .lifecycle import Lifecycle

from core.orchestration.brain import Brain

from ..memory.memory_manager import MemoryManager
from ..memory.memory_consolidator import (
    MemoryConsolidator,
)

from ..learning.experience_engine import (
    ExperienceEngine,
)
from ..learning.self_evaluator import (
    SelfEvaluator,
)
from ..learning.knowledge_builder import (
    KnowledgeBuilder,
)
from ..learning.learning_coordinator import (
    LearningCoordinator,
)
from ..learning.evolution_engine import (
    EvolutionEngine,
)


# =============================================================
# CREATE JARVIS
# =============================================================

def create_jarvis(
    identity=None,
    personality=None,
    values=None,
    heartbeat_interval: float = 5.0,
    idle_threshold: float = 30.0,
) -> JarvisCore:
    """
    Construct the complete JARVIS organism.

    Dependency architecture:

        InternalState
              │
              ▼
          EventBus
              │
        ┌─────┴─────┐
        ▼           ▼
     Heartbeat    Memory
                    │
          ┌─────────┼───────────┐
          ▼         ▼           ▼
     Experience  Evaluator   Knowledge
       Engine                  Builder
          │         │           │
          └─────────┼───────────┘
                    ▼
          MemoryConsolidator
                    │
                    ▼
          LearningCoordinator
                    │
                    ▼
            EvolutionEngine
                    │
                    ▼
                  Brain
                    │
                    ▼
               JarvisCore
                    │
                    ▼
                Lifecycle
    """

    # =========================================================
    # 1. INTERNAL STATE
    # =========================================================

    state = InternalState()

    # =========================================================
    # 2. EVENT BUS
    # =========================================================

    events = EventBus(
        internal_state=state,
    )

    # =========================================================
    # 3. HEARTBEAT
    # =========================================================

    heartbeat = Heartbeat(
        event_bus=events,
        internal_state=state,
        interval=heartbeat_interval,
        idle_threshold=idle_threshold,
    )

    # =========================================================
    # 4. MEMORY
    # =========================================================

    memory = MemoryManager(
        event_bus=events,
    )

    # =========================================================
    # 5. EXPERIENCE ENGINE
    # =========================================================
    #
    # Converts raw completed experiences into structured
    # experiences and learning signals.
    #

    experience_engine = ExperienceEngine(
        memory_manager=memory,
        event_bus=events,
        internal_state=state,
    )

    # =========================================================
    # 6. SELF EVALUATOR
    # =========================================================
    #
    # Evaluates whether an experience was useful,
    # successful, reliable, etc.
    #

    evaluator = SelfEvaluator(
        memory_manager=memory,
        event_bus=events,
        internal_state=state,
    )

    # =========================================================
    # 7. KNOWLEDGE BUILDER
    # =========================================================
    #
    # Converts evaluated experiences into knowledge
    # candidates.
    #

    knowledge_builder = KnowledgeBuilder(
        event_bus=events,
        internal_state=state,
        memory_manager=memory,
    )

    # =========================================================
    # 8. MEMORY CONSOLIDATOR
    # =========================================================
    #
    # Converts important/repeated episodic experiences
    # into semantic knowledge.
    #

    consolidator = MemoryConsolidator(
        memory_manager=memory,
        event_bus=events,
    )

    # =========================================================
    # 9. LEARNING COORDINATOR
    # =========================================================
    #
    # Central learning orchestration layer.
    #
    # It does NOT replace the organs.
    #
    # It coordinates:
    #
    #     Experience
    #          ↓
    #     Evaluation
    #          ↓
    #     Knowledge
    #          ↓
    #     Acceptance
    #          ↓
    #     Consolidation
    #

    learning = LearningCoordinator(
        evaluator=evaluator,
        knowledge_builder=knowledge_builder,
        consolidator=consolidator,
        memory_manager=memory,
        event_bus=events,
        internal_state=state,
    )

    # =========================================================
    # 10. EVOLUTION ENGINE
    # =========================================================
    #
    # Evolution remains completely controlled.
    #
    #     Proposal
    #        ↓
    #     Validate
    #        ↓
    #      Approve
    #        ↓
    #       Apply
    #

    evolution = EvolutionEngine(
        event_bus=events,
        internal_state=state,
        memory_manager=memory,
    )

    # =========================================================
    # 11. BRAIN
    # =========================================================
    #
    # Brain is the high-level orchestrator.
    #
    # Brain does NOT duplicate learning logic.
    #

    brain = Brain(
        memory_manager=memory,
        experience_engine=experience_engine,
        learning_coordinator=learning,
        self_evaluator=evaluator,
        knowledge_builder=knowledge_builder,
        memory_consolidator=consolidator,
        evolution_engine=evolution,
        event_bus=events,
        internal_state=state,
    )

    # =========================================================
    # 12. JARVIS CORE
    # =========================================================

    jarvis = JarvisCore(
        identity=identity,
        personality=personality,
        values=values,
        state=state,
        event_bus=events,
        heartbeat=heartbeat,
    )

    # =========================================================
    # 13. ATTACH ORGANS
    # =========================================================

    jarvis.attach_organ(
        "memory",
        memory,
    )

    jarvis.attach_organ(
        "experience_engine",
        experience_engine,
    )

    jarvis.attach_organ(
        "self_evaluator",
        evaluator,
    )

    jarvis.attach_organ(
        "knowledge_builder",
        knowledge_builder,
    )

    jarvis.attach_organ(
        "memory_consolidator",
        consolidator,
    )

    jarvis.attach_organ(
        "learning_coordinator",
        learning,
    )

    jarvis.attach_organ(
        "evolution",
        evolution,
    )

    jarvis.attach_organ(
        "brain",
        brain,
    )

    # =========================================================
    # 14. LIFECYCLE
    # =========================================================

    lifecycle = Lifecycle(
        jarvis=jarvis,
        internal_state=state,
        event_bus=events,
        heartbeat=heartbeat,
    )

    jarvis.lifecycle = lifecycle

    # =========================================================
    # 15. FINAL RUNTIME REFERENCES
    # =========================================================
    #
    # Optional direct references on JarvisCore.
    # Only assign them if the attributes already exist or
    # the project architecture allows dynamic attributes.
    #

    jarvis.brain = brain

    return jarvis


# =============================================================
# START JARVIS
# =============================================================

def start_jarvis(
    identity=None,
    personality=None,
    values=None,
    heartbeat_interval: float = 5.0,
    idle_threshold: float = 30.0,
) -> JarvisCore:
    """
    Create and start JARVIS.
    """

    jarvis = create_jarvis(
        identity=identity,
        personality=personality,
        values=values,
        heartbeat_interval=heartbeat_interval,
        idle_threshold=idle_threshold,
    )

    lifecycle = getattr(
        jarvis,
        "lifecycle",
        None,
    )

    if lifecycle is None:
        raise RuntimeError(
            "JARVIS lifecycle is not connected."
        )

    success = lifecycle.start()

    if not success:
        raise RuntimeError(
            "JARVIS failed to start."
        )

    jarvis.running = True

    return jarvis


# =============================================================
# STOP JARVIS
# =============================================================

def stop_jarvis(
    jarvis: Optional[JarvisCore],
) -> None:
    """
    Safely stop JARVIS.

    Shutdown order:

        Lifecycle
            ↓
        Heartbeat
            ↓
        Memory
            ↓
        EventBus
    """

    if jarvis is None:
        return

    # =========================================================
    # 1. LIFECYCLE
    # =========================================================

    lifecycle = getattr(
        jarvis,
        "lifecycle",
        None,
    )

    if lifecycle is not None:

        stop_method = getattr(
            lifecycle,
            "stop",
            None,
        )

        if callable(stop_method):

            stop_method()

    # =========================================================
    # 2. BRAIN
    # =========================================================

    brain = getattr(
        jarvis,
        "brain",
        None,
    )

    if brain is not None:

        stop_method = getattr(
            brain,
            "stop",
            None,
        )

        if callable(stop_method):

            stop_method()

    # =========================================================
    # 3. LEARNING COORDINATOR
    # =========================================================

    learning = None

    try:

        learning = jarvis.get_organ(
            "learning_coordinator"
        )

    except Exception:
        learning = None

    if learning is not None:

        stop_method = getattr(
            learning,
            "stop",
            None,
        )

        if callable(stop_method):

            stop_method()

    # =========================================================
    # 4. MEMORY
    # =========================================================

    memory = None

    try:

        memory = jarvis.get_organ(
            "memory"
        )

    except Exception:
        memory = None

    if memory is not None:

        close_method = getattr(
            memory,
            "close",
            None,
        )

        if callable(close_method):

            close_method()

    # =========================================================
    # 5. EVENT BUS
    # =========================================================

    events = getattr(
        jarvis,
        "events",
        None,
    )

    if events is not None:

        stop_method = getattr(
            events,
            "stop",
            None,
        )

        if callable(stop_method):

            stop_method()

    # =========================================================
    # 6. FINAL STATE
    # =========================================================

    jarvis.running = False