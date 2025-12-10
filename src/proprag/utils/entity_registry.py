import json
import os
import logging
from typing import Dict, List, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)


class GlobalEntityRegistry:
    def __init__(self, llm_model: Any, save_path: Optional[str] = None):
        self.registry: Dict[str, Dict[str, Any]] = {}  # Canonical Name -> {profile, aliases}
        self.reverse_lookup: Dict[str, str] = {}  # Alias -> Canonical Name

        self.recent_entities = deque(maxlen=10)

        self.llm = llm_model
        self.save_path = save_path
        if self.save_path and os.path.exists(self.save_path):
            self._load()
        logger.info(f"GlobalEntityRegistry initialized with {len(self.registry)} entries.")

    def _load(self):
        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.registry = data.get("registry", {})
                self.reverse_lookup = data.get("reverse_lookup", {})
            logger.info(f"Loaded registry from {self.save_path}")
        except Exception as e:
            logger.error(f"Failed to load GlobalEntityRegistry: {e}")
            raise

    def _save(self):
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump({"registry": self.registry, "reverse_lookup": self.reverse_lookup}, f, ensure_ascii=False,
                          indent=2)

    def resolve_and_add(self, candidate_entity_name: str, context_text: str, chunk_id: str) -> str:
        """Synchronously resolves or adds an entity."""
        assert candidate_entity_name and isinstance(candidate_entity_name,
                                                    str), "Candidate entity name must be a non-empty string."

        candidate_entity_name = candidate_entity_name.strip()
        if not candidate_entity_name: return ""

        if candidate_entity_name in self.reverse_lookup:
            canonical = self.reverse_lookup[candidate_entity_name]
            # [Rashomon Fix] 即使查到了，也要更新“最近活跃”，保持上下文新鲜
            if canonical not in self.recent_entities:
                self.recent_entities.appendleft(canonical)
            return canonical

        # Use LLM for resolution
        known_entities_lines = []
        for name, data in self.registry.items():
            aliases = data.get("aliases", [])
            aliases_str = f" (Aliases: {', '.join(aliases)})" if aliases else ""
            known_entities_lines.append(f"- {name}{aliases_str}: {data.get('profile', '')}")

        known_entities_str = "\n".join(known_entities_lines) or "None"

        # B. 短期记忆 (Working Memory) - 关键修改
        # 将 deque 转为字符串列表，强调顺序 (越靠前越近)
        recent_context_list = list(self.recent_entities)
        recent_context_str = ", ".join(recent_context_list) if recent_context_list else "None"

        system_prompt = """You are an Entity Resolver with Context Awareness. 
        Your task is to resolve an entity mention in a text to its Canonical Name.

        Priority of Resolution:
        1. **Context Match**: If the mention refers to someone in the 'Recently Active Entities' list (e.g., 'the source' -> 'Anonymous Source'), mapping them is the HIGHEST priority.
        2. **Global Match**: If not recent, check the 'Global Known Entities'.
        3. **New Entity**: If strictly new, create a new canonical name.

        Respond in JSON: {"canonical_name": "...", "is_new": bool, "profile": "..."}"""

        user_prompt = f"""
        Current Text Chunk: "{context_text}"

        [CRITICAL CONTEXT] Recently Active Entities (mentioned in previous chunks):
        [{recent_context_str}]

        [Knowledge Base] Global Known Entities:
        {known_entities_str}

        Task: Resolve the entity mention "{candidate_entity_name}".

        Thinking Process:
        1. Is "{candidate_entity_name}" a pronoun or reference (e.g., "he", "the source", "the plan")?
        2. If YES, does it likely refer to someone in [Recently Active Entities]?
        3. If NO, is it a variation of a [Global Known Entity]?

        Output JSON:
        """

        try:
            raw_response, _, _ = self.llm.infer(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.0, response_format={"type": "json_object"}
            )
            response_data = json.loads(raw_response)

            canonical_name = response_data.get("canonical_name", candidate_entity_name).strip()
            assert canonical_name and isinstance(canonical_name, str), \
                f"LLM resolver error: 'canonical_name' must be a non-empty string. Got: {canonical_name}"

            is_new = response_data.get("is_new", True)
            profile = response_data.get("profile", "")

            # Update registry
            if is_new and canonical_name not in self.registry:
                self.registry[canonical_name] = {
                    "profile": profile,
                    "aliases": [candidate_entity_name],
                    "first_seen": chunk_id}
                logger.info(f"Registered new entity: {canonical_name}")
            elif canonical_name in self.registry and candidate_entity_name not in self.reverse_lookup:
                # If it's an existing canonical name but a new alias, add the alias
                if candidate_entity_name not in self.registry[canonical_name].get("aliases", []):
                    self.registry[canonical_name]["aliases"].append(candidate_entity_name)

            # 5. 更新反向查找表
            self.reverse_lookup[candidate_entity_name] = canonical_name
            # 确保 Canonical Name 自己也能查到自己
            if canonical_name not in self.reverse_lookup:
                self.reverse_lookup[canonical_name] = canonical_name

            # 6. [Rashomon Fix] 更新短期记忆
            # 将解析出的结果放入滑动窗口头部
            if canonical_name in self.recent_entities:
                self.recent_entities.remove(canonical_name)  # 移出旧位置
            self.recent_entities.appendleft(canonical_name)  # 放到最前

            self._save()
            return canonical_name

        except Exception as e:
            logger.error(f"LLM resolution failed for '{candidate_entity_name}': {e}. Treating as new.")
            assert candidate_entity_name and isinstance(candidate_entity_name, str), \
                "Fallback error: candidate_entity_name is somehow invalid."

            if candidate_entity_name not in self.registry:
                self.registry[candidate_entity_name] = {"profile": "Auto-registered due to error.",
                                                        "aliases": [candidate_entity_name]}
            self.reverse_lookup[candidate_entity_name] = candidate_entity_name
            self.recent_entities.appendleft(candidate_entity_name)
            return candidate_entity_name

    def get_known_entities_for_prompt(self, max_tokens=1000) -> str:
        lines = []
        for name, data in self.registry.items():
            aliases = data.get("aliases", [])
            aliases_str = f" (aka: {', '.join(aliases)})" if aliases else ""

            entry = f"- {name}{aliases_str}: {data.get('profile', 'No profile.')}\n"
            lines.append(entry)
        return "".join(lines) or "No entities registered yet."
