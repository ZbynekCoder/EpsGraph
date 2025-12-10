import json
import logging
from typing import List, Dict, Any, Optional

# from ..llm.base import BaseLLM # 如果需要直接传 LLM 实例
from src.proprag.llm.openai_gpt import CacheOpenAI  # 假设你用的是 CacheOpenAI

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
        memory_context = ""
        if not retrieved_memories:
            memory_context = f"Agent {agent_name} has no directly related prior beliefs on record."
        else:
            memory_context = f"Agent {agent_name}'s known prior beliefs (related to '{new_belief_text[:50]}...'):\n"
            for i, p in enumerate(retrieved_memories):
                # 提取路径中 Belief 节点的文本，忽略 Agent/Entity 节点
                belief_texts = [text for j, text in enumerate(p['texts']) if p['nodes'][j].startswith("proposition-")]
                # 优先使用路径末端的 Belief，因为它与 query 最相关
                core_belief_text = belief_texts[-1] if belief_texts else "Unknown Belief"
                memory_context += f"- {core_belief_text} (Source: {p.get('source', agent_name)})\n"  # Path owner check

        # 2. 构建 Prompt
        system_prompt = f"""
        You are an AI Auditor. Your task is to analyze whether a new statement by Agent '{agent_name}' 
        is consistent with their previously recorded beliefs and persona.

        Agent Name: {agent_name}
        {f"Agent Persona: {agent_persona}" if agent_persona else ""}

        Evaluate the new statement against the agent's prior beliefs.

        Output a JSON object with two fields:
        - "status": "Consistent", "Inconsistent", or "Neutral" (if no clear conflict/support)
        - "reasoning": A detailed explanation of your judgment. If inconsistent, explain why.
        """

        user_prompt = f"""
        Prior Beliefs/Memory for Agent '{agent_name}':
        {memory_context}

        New Statement/Action by Agent '{agent_name}':
        "{new_belief_text}"

        Is this new statement consistent with {agent_name}'s prior beliefs/persona?
        """

        logger.debug(f"Calling LLM for consistency validation for agent {agent_name}.")
        logger.debug(f"Prompt:\n{user_prompt}")

        try:
            raw_response, metadata, _ = self.llm.infer(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # 确保判断结果稳定
                response_format={"type": "json_object"}
            )

            # 尝试解析 JSON
            response_data = json.loads(raw_response)
            status = response_data.get("status", "Error").strip()
            reasoning = response_data.get("reasoning", "Failed to parse reasoning.").strip()

        except Exception as e:
            logger.error(f"Error during consistency validation LLM call for {agent_name}: {e}")
            status = "Error"
            reasoning = f"LLM call failed: {e}"
            raw_response = {}  # Ensure raw_response is defined

        return {
            "agent": agent_name,
            "new_belief": new_belief_text,
            "status": status,
            "reasoning": reasoning,
            "llm_response_raw": raw_response  # 方便调试
        }

