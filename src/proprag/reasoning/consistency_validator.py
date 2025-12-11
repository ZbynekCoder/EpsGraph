import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConsistencyValidator:
    def __init__(self, llm_model: Any):  # 使用 Any 保持灵活性
        self.llm = llm_model

    def validate(self,
                 agent_name: str,
                 new_belief_text: str,
                 retrieved_memories: List[Dict[str, Any]],
                 agent_persona: Optional[str] = None) -> Dict[str, Any]:
        """
        Audits a new belief against an agent's existing, relevant memories.

        Args:
            agent_name: The canonical name of the agent being audited (e.g., "Donald Trump").
            new_belief_text: The text of the new belief/statement/action by the agent.
            retrieved_memories: A list of relevant belief paths from the agent's subgraph,
                                each formatted by BeamSearchPathFinder._format_paths.
            agent_persona: An optional string describing the agent's persona or typical behavior.

        Returns:
            A dictionary containing the validation result:
            - "status": "Consistent", "Inconsistent", "Neutral"
            - "reasoning": An explanation for the status.
        """

        # 1. 格式化 Agent 的记忆
        memory_txt = "\n".join(
            [f"- {m['text']} (Source: {m['source']})" for m in retrieved_memories]) if retrieved_memories else "None."
        persona_txt = agent_persona if agent_persona else "Unknown Persona."

        system_prompt = f"""
                You are an AI Auditor analyzing Cognitive Consistency.
                Target Agent: '{agent_name}'

                Tasks:
                1. Check consistency with **Persona** (Traits): Is the statement Out-Of-Character?
                2. Check consistency with **Memory** (Prior Beliefs): Does it contradict past statements?

                Output JSON: {{"status": "Consistent"|"Inconsistent"|"Neutral", "reasoning": "..."}}
                """

        user_prompt = f"""
                [AGENT PROFILE]:
                {persona_txt}

                [RELEVANT MEMORIES]:
                {memory_txt}

                [NEW STATEMENT]:
                "{new_belief_text}"

                Audit this statement.
                """

        try:
            raw, _, _ = self.llm.infer(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.0, response_format={"type": "json_object"}
            )
            data = json.loads(raw)
            return {
                "status": data.get("status", "Error"),
                "reasoning": data.get("reasoning", "Parse Error")
            }
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"status": "Error", "reasoning": str(e)}

