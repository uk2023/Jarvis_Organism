from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional


class Brain:
    """
    Central orchestration organ of JARVIS.

    Brain coordinates major cognitive organs.

    Brain is NOT:
        - the LLM
        - the memory database
        - the learning engine itself
        - the evaluator
        - the knowledge builder
        - the evolution engine
        - an unrestricted executor

    Architecture:

        Input
          ↓
        Brain
          ↓
        ExperienceEngine
          ↓
        LearningCoordinator
          ↓
        SelfEvaluator
          ↓
        KnowledgeBuilder
          ↓
        Memory / Consolidation

    Evolution remains controlled:

        Proposal
            ↓
        Validate
            ↓
        Approve
            ↓
        Apply

    Brain only orchestrates these operations.

    ----------------------------------------------------------------
    FIX LOG (this version)
    ----------------------------------------------------------------
    Root cause of "knowledge table stays at 0 rows even though
    episodes/chat_messages keep growing":

        process_experience() was always building a knowledge
        CANDIDATE (via KnowledgeBuilder.build / LearningCoordinator.learn)
        but never ACCEPTING it. auto_accept defaulted to False
        everywhere it was called from (including think_and_respond),
        and the compatibility-fallback branch hard-coded
        "accepted": False with no accept step at all.

        A built-but-unaccepted knowledge candidate is normally kept
        out of the persistent knowledge store on purpose (human/agent
        review gate) — but nothing in the pipeline was ever calling
        accept_knowledge() afterwards, so candidates just evaporated.

    Fix:
        1. Brain now has self.auto_accept_knowledge (default True).
           think_and_respond() and process_experience() use this
           instead of a hard-coded False.
        2. The compatibility-fallback branch (no LearningCoordinator
           connected) now actually calls knowledge_builder.accept()
           when auto_accept is True, instead of silently discarding
           the candidate.
        3. Removed dead/unreachable code after the first `return` in
           think_and_respond (duplicate pipeline-trigger block and a
           duplicate `except` that could never execute and would have
           been a SyntaxError-adjacent trap).
        4. Removed the duplicate `_finish_cycle` / `_emit` method
           definitions (Python just silently used the second one,
           but keeping two definitions of the same method is a
           landmine for future edits).

    5. NEW: even with (1)-(4) fixed, normal chat still produced
       zero knowledge. KnowledgeBuilder._extract_semantic_fact()
       only accepts an EXPLICIT subject/predicate/value triple
       already present in `context` or `outcome` — free chat text
       never has that shape on its own. think_and_respond() now
       makes a second, small LLM call (_extract_fact) after every
       turn to pull a {subject, predicate, value} triple out of
       the conversation (if one exists) and merges it into
       `outcome` before process_experience() runs. This is the
       step that actually turns "meri ex ka naam Devyana hai" into
       a rememberable fact instead of just a chat log line.
    ----------------------------------------------------------------
    """

    VERSION = "0.5.0"

    def __init__(
        self,
        memory_manager=None,
        experience_engine=None,
        self_evaluator=None,
        knowledge_builder=None,
        memory_consolidator=None,
        learning_coordinator=None,
        evolution_engine=None,
        event_bus=None,
        internal_state=None,
        planner=None,
        goal_manager=None,
        llm_bridge=None,
        auto_accept_knowledge: bool = True,
    ):
        # =========================================================
        # CORE ORGANS
        # =========================================================

        self.memory = memory_manager
        self.experience = experience_engine
        self.evaluator = self_evaluator
        self.knowledge_builder = knowledge_builder
        self.consolidator = memory_consolidator
        self.learning = learning_coordinator
        self.evolution = evolution_engine

        # =========================================================
        # SYSTEM SERVICES
        # =========================================================

        self.events = event_bus
        self.state = internal_state
        self.planner = planner
        self.goal_manager = goal_manager
        self.llm = llm_bridge

        # =========================================================
        # LEARNING POLICY
        # =========================================================
        # Whether experiences that pass through process_experience()
        # get their resulting knowledge candidate auto-accepted into
        # persistent memory. This is what actually makes JARVIS learn
        # instead of just logging episodes. Set to False if you want
        # a manual review step (accept_knowledge / reject_knowledge).
        self.auto_accept_knowledge = auto_accept_knowledge

        # =========================================================
        # RUNTIME
        # =========================================================

        self.created_at = time.time()
        self.last_cycle_at: Optional[float] = None
        self.cycle_count = 0
        self.last_result: Optional[Dict[str, Any]] = None
        self.running = True

    # =============================================================
    # THINK AND RESPOND (LLM + IDENTITY + MEMORY + PIPELINE)
    # =============================================================
    
    def think_and_respond(
        self,
        user_input: str,
        identity_profile: Optional[Dict[str, Any]] = None,
        source: str = "cli",
    ) -> str:
    
        """
        Synthesizes Identity, Memory Context, and User Input,
        queries the LLM bridge (used purely as a voice — see
        llm_bridge.py), and passes the resulting experience through
        the full Organism Learning Pipeline so JARVIS actually
        retains it instead of just generating a reply.
        """
        if self.llm is None:
            return "[Brain Error: LLM Bridge is not connected to Brain.]"

        # 1. Build Context from FAISS Vector Store + Knowledge Graph
        context = (
            self.build_context(query=user_input, recent_limit=3)
            if hasattr(self, "build_context")
            else {}
        )
        recent_memories = context.get("recent_experiences", [])
        relevant_knowledge = context.get("relevant_knowledge", [])
        graph_relations = context.get("graph_relations", [])

        # 2. Parse Identity Info
        bot_name = "JARVIS"
        creator_name = "UK"
        if isinstance(identity_profile, dict):
            bot_name = identity_profile.get("name", bot_name)
            creator_name = identity_profile.get("creator", creator_name)

        # 3. Strict Identity System Prompt
        system_prompt = (
            f"SYSTEM IDENTITY INSTRUCTIONS (STRICT):\n"
            f"1. You are {bot_name}, a self-contained cognitive AI organism running locally.\n"
            f"2. The user talking to you is {creator_name}, your developer and creator.\n"
            f"3. NEVER swap roles or claim to be {creator_name}.\n"
            f"4. Communicate naturally, precisely, and conversationally in Hinglish.\n"
            f"5. Keep responses loyal, calm, and too short.\n"
            f"6. Dont use emojis in response.\n"
            f"7. Behave and act alike Marvel iron man's JARVIS and give response savagely and funny.\n"
            f"8. Always loyal to {creator_name}."
        )

        # 4. Context Formatting
        context_prompt = (
            f"=== RETRIEVED MEMORIES ===\n{recent_memories if recent_memories else 'No previous memory match.'}\n\n"
            f"=== SEMANTIC KNOWLEDGE ===\n{relevant_knowledge if relevant_knowledge else 'No direct facts found.'}\n\n"
            f"=== KNOWLEDGE GRAPH EDGES ===\n{graph_relations if graph_relations else 'No graph nodes linked.'}\n\n"
            f"=== CURRENT USER MESSAGE ===\n{creator_name}: {user_input}\n\n"
            f"{bot_name}:"
        )

        try:
            # 5. LLM Inference (voice only — no memory logic lives here)
            response = self.llm.generate_response(
                system_prompt=system_prompt, user_input=context_prompt
            )
            cleaned_response = (
                response.strip() if isinstance(response, str) else str(response)
            )
        except Exception as exc:
            return f"[Brain Thinking Error: {exc}]"

        # 6. Push the interaction through the full learning pipeline
        #    so it actually becomes persistent knowledge, not just a
        #    chat log line. This is the step that was previously
        #    silently failing to persist anything.
        #
        #    6a. KnowledgeBuilder needs an explicit subject/predicate/
        #    value triple inside `outcome` — raw chat text never has
        #    that shape on its own, so without this step NOTHING said
        #    in normal conversation could ever become knowledge.
        outcome: Dict[str, Any] = {"status": "completed"}
        fact = self._extract_fact(user_input, cleaned_response)
        if fact is not None:
            outcome.update(fact)  # adds subject/predicate/value

        if hasattr(self, "process_experience"):
            try:
                self.process_experience(
                    event_type="USER_CHAT",
                    context={"user_input": user_input},
                    action={"jarvis_response": cleaned_response},
                    outcome=outcome,
                    source=source,
                    importance=0.6,
                    build_knowledge=True,
                    auto_accept=self.auto_accept_knowledge,
                )
            except Exception as exp_err:
                print(f"[Brain Pipeline Warning] Could not process experience: {exp_err}")
        elif (
            hasattr(self, "memory")
            and self.memory
            and hasattr(self.memory, "remember_experience")
        ):
            self.memory.remember_experience(
                event_type="USER_CHAT",
                context={"user_input": user_input},
                action={"jarvis_response": cleaned_response},
                importance=0.6,
            )

        return cleaned_response

    # =============================================================
    # FACT EXTRACTION (turns free chat into a structured triple)
    # =============================================================

    _FACT_EXTRACTION_PROMPT = (
        "You are an expert fact extractor for a personal AI companion. "
        "Extract ONE factual statement from the conversation turn if one exists. "
        "The user often speaks in Hinglish with minor spelling typos (e.g. 'nan' instead of 'naam'). "
        "Correct typos automatically and extract clear subject, predicate, and value. "
        "Return ONLY a raw JSON object. If no clear fact exists, return exactly: "
        '{"has_fact": false}\n\n'
        "Examples:\n"
        'User: "mera ex ka nan devyana h"\n'
        'Output: {"has_fact": true, "subject": "user_ex", "predicate": "name", "value": "Devyana"}\n\n'
        "subject/predicate should be short lowercase phrases."      
    )

    def _extract_fact(self, user_input: str, jarvis_response: str) -> Optional[Dict[str, Any]]:
        if self.llm is None:
            return None

        # Thoda sa gap dein taaki key rotator next active key pick kar sake
        import time
        time.sleep(1.0)

        raw = ""
        try:
            raw = self.llm.generate_response(
                system_prompt=getattr(self, "_FACT_EXTRACTION_PROMPT", "Extract facts as JSON with subject, predicate, value."),
                user_input=f"User said: {user_input}\nAssistant replied: {jarvis_response}",
                max_tokens=500,
                temperature=0.0,
            )
        except Exception as e:
            print(f"[EXTRACT ERROR WITH KEYS]: {e}")

        print(f"[DEBUG ROTATED KEY FACT OUTPUT]: {repr(raw)}")

        if not raw or not isinstance(raw, str) or not raw.strip():
            return None

        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

        try:
            data = json.loads(cleaned)
        except Exception:
            return None

        if not isinstance(data, dict) or not data.get("has_fact"):
            return None

        subject = str(data.get("subject", "")).strip()
        predicate = str(data.get("predicate", "")).strip()
        value = data.get("value")

        if not subject or not predicate or value in (None, ""):
            return None

        print(f"[SUCCESS EXTRACTED]: {subject} -> {predicate} -> {value}")
        return {"subject": subject, "predicate": predicate, "value": value}



    # =============================================================
    # PROCESS EXPERIENCE
    # =============================================================

    def process_experience(
        self,
        event_type: str,
        context: Optional[Dict[str, Any]] = None,
        action: Optional[Dict[str, Any]] = None,
        outcome: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        importance: float = 0.5,
        build_knowledge: bool = True,
        auto_accept: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Process one completed experience.

        Flow:
            Brain -> ExperienceEngine -> LearningCoordinator ->
            SelfEvaluator -> KnowledgeBuilder -> acceptance

        `auto_accept` defaults to self.auto_accept_knowledge when not
        explicitly passed, so the "built but never persisted" bug
        can't silently recur.
        """
        if self.experience is None:
            raise RuntimeError("ExperienceEngine is not connected.")

        if auto_accept is None:
            auto_accept = self.auto_accept_knowledge

        started_at = time.time()

        # =========================================================
        # 1. EXPERIENCE ENGINE
        # =========================================================
        experience_result = self.experience.process(
            event_type=event_type,
            context=context or {},
            action=action or {},
            outcome=outcome or {},
            source=source,
            importance=importance,
        )

        if not isinstance(experience_result, dict):
            raise RuntimeError("ExperienceEngine returned an invalid result.")

        experience = experience_result.get("experience", {})

        # =========================================================
        # 2. LEARNING COORDINATOR (preferred path)
        # =========================================================
        learning_result = None

        if self.learning is not None and build_knowledge:
            learn_method = getattr(self.learning, "learn", None)
            if not callable(learn_method):
                raise RuntimeError("LearningCoordinator does not expose learn().")

            learning_result = learn_method(
                experience=experience,
                auto_accept=auto_accept,
            )

        # =========================================================
        # 3. COMPATIBILITY FALLBACK (no LearningCoordinator connected)
        # =========================================================
        elif self.learning is None and build_knowledge:
            evaluation = None
            if self.evaluator is not None:
                evaluation = self.evaluator.evaluate(experience)

            knowledge = None
            if self.knowledge_builder is not None and evaluation is not None:
                knowledge = self.knowledge_builder.build(
                    experience=experience,
                    evaluation=evaluation,
                )

            accepted = False
            # THIS is the part that was previously missing: without
            # it, `knowledge` was created but never written to the
            # persistent knowledge table.
            if (
                auto_accept
                and knowledge is not None
                and self.knowledge_builder is not None
            ):
                knowledge_id = (
                    knowledge.get("id")
                    if isinstance(knowledge, dict)
                    else getattr(knowledge, "id", None)
                )
                accept_method = getattr(self.knowledge_builder, "accept", None)
                if knowledge_id is not None and callable(accept_method):
                    try:
                        accept_method(knowledge_id)
                        accepted = True
                    except Exception as accept_err:
                        print(f"[Brain Pipeline Warning] Could not auto-accept knowledge: {accept_err}")

            learning_result = {
                "success": True,
                "experience": experience,
                "evaluation": evaluation,
                "knowledge": knowledge,
                "accepted": accepted,
                "duration": 0.0,
                "timestamp": time.time(),
            }

        # =========================================================
        # 4. BUILD RESULT
        # =========================================================
        evaluation = None
        knowledge = None
        accepted = False

        if isinstance(learning_result, dict):
            evaluation = learning_result.get("evaluation")
            knowledge = learning_result.get("knowledge")
            accepted = bool(learning_result.get("accepted", False))

        result = {
            "type": "BRAIN_EXPERIENCE_CYCLE",
            "success": True,
            "experience": experience,
            "learning": learning_result,
            "evaluation": evaluation,
            "knowledge": knowledge,
            "accepted": accepted,
            "episode_id": experience_result.get("episode_id"),
            "duration": time.time() - started_at,
            "timestamp": time.time(),
        }

        self._finish_cycle(result)
        self._emit("BRAIN_EXPERIENCE_PROCESSED", result)

        return result

    # =============================================================
    # LEARN
    # =============================================================

    def learn(self, experience: Dict[str, Any], auto_accept: Optional[bool] = None) -> Dict[str, Any]:
        """Direct learning entry point for an already-structured experience."""
        if not isinstance(experience, dict):
            raise TypeError("experience must be a dictionary.")

        if self.learning is None:
            raise RuntimeError("LearningCoordinator is not connected.")

        method = getattr(self.learning, "learn", None)
        if not callable(method):
            raise RuntimeError("LearningCoordinator does not expose learn().")

        if auto_accept is None:
            auto_accept = self.auto_accept_knowledge

        result = method(experience=experience, auto_accept=auto_accept)
        self._emit("BRAIN_LEARNING_COMPLETED", result)
        return result

    # =============================================================
    # EVALUATE
    # =============================================================

    def evaluate(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        if self.learning is not None:
            method = getattr(self.learning, "evaluate", None)
            if callable(method):
                return method(experience)

        if self.evaluator is None:
            raise RuntimeError("SelfEvaluator is not connected.")

        return self.evaluator.evaluate(experience)

    # =============================================================
    # BUILD KNOWLEDGE
    # =============================================================

    def build_knowledge(
        self,
        experience: Dict[str, Any],
        evaluation: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.learning is not None:
            method = getattr(self.learning, "build_knowledge", None)
            if callable(method):
                return method(experience=experience, evaluation=evaluation)

        if self.knowledge_builder is None:
            raise RuntimeError("KnowledgeBuilder is not connected.")

        if evaluation is None:
            if self.evaluator is None:
                raise RuntimeError("SelfEvaluator is not connected.")
            evaluation = self.evaluator.evaluate(experience)

        return self.knowledge_builder.build(experience=experience, evaluation=evaluation)

    # =============================================================
    # ACCEPT / REJECT KNOWLEDGE
    # =============================================================

    def accept_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        if self.learning is not None:
            method = getattr(self.learning, "accept_knowledge", None)
            if callable(method):
                result = method(knowledge_id)
                self._emit("BRAIN_KNOWLEDGE_ACCEPTED", result)
                return result

        if self.knowledge_builder is None:
            raise RuntimeError("KnowledgeBuilder is not connected.")

        result = self.knowledge_builder.accept(knowledge_id)
        self._emit("BRAIN_KNOWLEDGE_ACCEPTED", result)
        return result

    def reject_knowledge(self, knowledge_id: str, reason: str = "") -> Optional[Dict[str, Any]]:
        if self.learning is not None:
            method = getattr(self.learning, "reject_knowledge", None)
            if callable(method):
                result = method(knowledge_id=knowledge_id, reason=reason)
                self._emit("BRAIN_KNOWLEDGE_REJECTED", result)
                return result

        if self.knowledge_builder is None:
            raise RuntimeError("KnowledgeBuilder is not connected.")

        result = self.knowledge_builder.reject(knowledge_id=knowledge_id, reason=reason)
        self._emit("BRAIN_KNOWLEDGE_REJECTED", result)
        return result

    # =============================================================
    # CONSOLIDATE
    # =============================================================

    def consolidate(self, limit: int = 50) -> Dict[str, Any]:
        if self.consolidator is None:
            raise RuntimeError("MemoryConsolidator is not connected.")

        result = self.consolidator.consolidate(limit=limit)
        self._emit("BRAIN_MEMORY_CONSOLIDATED", result)
        return result

    def learn_and_consolidate(
        self,
        experience: Dict[str, Any],
        auto_accept: Optional[bool] = None,
        consolidation_limit: int = 50,
    ) -> Dict[str, Any]:
        learning_result = self.learn(experience=experience, auto_accept=auto_accept)

        consolidation_result = None
        if self.consolidator is not None:
            consolidation_result = self.consolidate(limit=consolidation_limit)

        return {
            "learning": learning_result,
            "consolidation": consolidation_result,
            "timestamp": time.time(),
        }

    # =============================================================
    # EVOLUTION
    # =============================================================

    def propose_evolution(self, evaluation: Dict[str, Any], target: str, reason: Optional[str] = None) -> Dict[str, Any]:
        if self.evolution is None:
            raise RuntimeError("EvolutionEngine is not connected.")
        proposal = self.evolution.propose(evaluation=evaluation, target=target, reason=reason)
        self._emit("BRAIN_EVOLUTION_PROPOSED", proposal)
        return proposal

    def validate_evolution(self, proposal_id: str) -> Dict[str, Any]:
        if self.evolution is None:
            raise RuntimeError("EvolutionEngine is not connected.")
        return self.evolution.validate(proposal_id)

    def approve_evolution(self, proposal_id: str) -> Dict[str, Any]:
        if self.evolution is None:
            raise RuntimeError("EvolutionEngine is not connected.")
        return self.evolution.approve(proposal_id)

    def apply_evolution(self, proposal_id: str) -> Dict[str, Any]:
        if self.evolution is None:
            raise RuntimeError("EvolutionEngine is not connected.")
        return self.evolution.apply(proposal_id)

    # =============================================================
    # MEMORY CONTEXT (FAISS + Knowledge Graph retrieval)
    # =============================================================

    def build_context(
        self,
        query: Optional[str] = None,
        subject: Optional[str] = None,
        recent_limit: int = 5,
        knowledge_limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Retrieve memory context for reasoning from MemoryManager.
        Brain intentionally does not know HOW retrieval works (FAISS
        similarity search, graph traversal, etc.) — that all lives in
        MemoryManager so it can be upgraded independently.
        """
        if self.memory is None:
            return {
                "recent_experiences": [],
                "relevant_knowledge": [],
                "graph_relations": [],
            }

        return self.memory.build_context(
            query=query,
            subject=subject,
            recent_limit=recent_limit,
            knowledge_limit=knowledge_limit,
        )

    # =============================================================
    # PLAN / GOALS
    # =============================================================

    def plan(self, goal: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        if self.planner is None:
            raise RuntimeError("Planner is not connected.")
        method = getattr(self.planner, "plan", None)
        if not callable(method):
            raise RuntimeError("Connected planner does not expose plan().")
        return method(goal=goal, context=context or {})

    def create_goal(self, goal: Any) -> Any:
        if self.goal_manager is None:
            raise RuntimeError("GoalManager is not connected.")
        method = getattr(self.goal_manager, "create_goal", None)
        if not callable(method):
            raise RuntimeError("Connected GoalManager does not expose create_goal().")
        return method(goal)

    # =============================================================
    # STATUS
    # =============================================================

    def status(self) -> Dict[str, Any]:
        learning_status = None
        if self.learning is not None:
            method = getattr(self.learning, "status", None)
            if callable(method):
                try:
                    learning_status = method()
                except Exception as exc:
                    learning_status = {"error": str(exc)}

        consolidator_status = None
        if self.consolidator is not None:
            method = getattr(self.consolidator, "status", None)
            if callable(method):
                try:
                    consolidator_status = method()
                except Exception as exc:
                    consolidator_status = {"error": str(exc)}

        return {
            "version": self.VERSION,
            "running": self.running,
            "created_at": self.created_at,
            "cycles": self.cycle_count,
            "last_cycle_at": self.last_cycle_at,
            "auto_accept_knowledge": self.auto_accept_knowledge,
            "organs": {
                "memory": self.memory is not None,
                "experience_engine": self.experience is not None,
                "self_evaluator": self.evaluator is not None,
                "knowledge_builder": self.knowledge_builder is not None,
                "memory_consolidator": self.consolidator is not None,
                "learning_coordinator": self.learning is not None,
                "evolution_engine": self.evolution is not None,
                "planner": self.planner is not None,
                "goal_manager": self.goal_manager is not None,
                "llm_bridge": self.llm is not None,
            },
            "learning_status": learning_status,
            "consolidator_status": consolidator_status,
        }

    def get_last_result(self) -> Optional[Dict[str, Any]]:
        return self.last_result

    # =============================================================
    # START / STOP
    # =============================================================

    def start(self) -> None:
        self.running = True
        if self.learning is not None:
            method = getattr(self.learning, "start", None)
            if callable(method):
                method()

    def stop(self) -> None:
        self.running = False
        if self.learning is not None:
            method = getattr(self.learning, "stop", None)
            if callable(method):
                method()

    # =============================================================
    # INTERNAL HELPERS
    # =============================================================

    def _finish_cycle(self, result: Dict[str, Any]) -> None:
        self.cycle_count += 1
        self.last_cycle_at = time.time()
        self.last_result = result

    def _emit(self, event_name: str, payload: Any = None) -> None:
        if self.events is None:
            return
        safe_emit = getattr(self.events, "safe_emit", None)
        if callable(safe_emit):
            safe_emit(event_name, payload, source="brain")
