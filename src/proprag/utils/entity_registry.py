import json
import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

import numpy as np

from src.proprag.utils.misc_utils import compute_mdhash_id

logger = logging.getLogger(__name__)


class GlobalEntityRegistry:
    def __init__(self, llm_model: Any, embedding_model: Any, entity_embedding_store: Any, save_path: Optional[str] = None):
        self.registry: Dict[str, Dict[str, Any]] = {}  # Canonical Name -> {profile, aliases}
        self.reverse_lookup: Dict[str, str] = {}  # Alias -> Canonical Name

        self.recent_entities = deque(maxlen=10)
        self.context_history = deque(maxlen=3)

        self.llm = llm_model
        self.embedding_model = embedding_model
        self.entity_embedding_store = entity_embedding_store
        self._ensure_embeddings_for_registry()

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

    def _ensure_embeddings_for_registry(self):
        canonical_names_to_embed = []
        for name in self.registry.keys():
            entity_hash_id = compute_mdhash_id(name, prefix="entity-")
            if entity_hash_id not in self.entity_embedding_store.hash_id_to_row:
                canonical_names_to_embed.append(name)

        if canonical_names_to_embed:
            logger.info(f"Embedding {len(canonical_names_to_embed)} new canonical entities for Registry.")
            self.entity_embedding_store.insert_strings(canonical_names_to_embed)

    def update_context_history(self, text: str):
        """将处理完的文本块加入历史缓存"""
        if text and isinstance(text, str):
            # 简单去重：如果和上一条不一样才加
            if not self.context_history or self.context_history[-1] != text:
                self.context_history.append(text)

    def resolve_and_add(self, candidate_entity_name: str, context_text: str, chunk_id: str, top_k_global_relevant: int = 5) -> str:
        """Synchronously resolves or adds an entity."""
        assert candidate_entity_name and isinstance(candidate_entity_name,
                                                    str), "Candidate entity name must be a non-empty string."

        candidate_entity_name = candidate_entity_name.strip()
        if not candidate_entity_name: return ""

        if candidate_entity_name in self.reverse_lookup:
            canonical = self.reverse_lookup[candidate_entity_name]
            if canonical not in self.recent_entities:
                self.recent_entities.appendleft(canonical)
            return canonical

        # Use LLM for resolution
        # 准备历史文本字符串
        if self.context_history:
            prev_context_str = "\n---\n".join(self.context_history)
        else:
            prev_context_str = "None (Start of story)"

        recent_active_entities_str, other_globally_relevant_entities_str = self.get_formatted_entities_for_prompt(
            candidate_entity_name=candidate_entity_name,
            top_k_global_relevant=top_k_global_relevant
        )

        system_prompt = """You are an Entity Resolver with Context Awareness. 
                Your task is to resolve an entity mention in a text to its Canonical Name.

                Priority of Resolution:
                1. **Context Match**: If the mention refers to someone in the 'Recently Active Entities' list (e.g., 'the source' -> 'Anonymous Source'), mapping them is the HIGHEST priority.
                2. **Global Match**: If not recent, check the 'Globally Relevant Known Entities'.
                3. **New Entity**: If strictly new, create a new canonical name.

                Respond in JSON: {"canonical_name": "...", "is_new": bool, "profile": "..."}"""

        user_prompt = f"""
                [PREVIOUS NARRATIVE CONTEXT] (What happened immediately before):
                \"\"\"
                {prev_context_str}
                \"\"\"

                [CURRENT TEXT CHUNK]: 
                "{context_text}"

                [CANDIDATE ENTITIES] (Recently active):
                {recent_active_entities_str}

                [GLOBAL KNOWLEDGE]:
                {other_globally_relevant_entities_str}

                Task: Resolve the entity mention "{candidate_entity_name}".

                Thinking Process:
                1. Look at [PREVIOUS NARRATIVE CONTEXT]. Who was the last person speaking or acting?
                2. If "{candidate_entity_name}" is "He"/"She", map it to that person.
                3. If strict match found in [CANDIDATE ENTITIES], use it.

                Output JSON: {{"canonical_name": "...", "is_new": bool, "profile": "..."}}
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

    def add_profile_trait(self, canonical_name: str, trait_text: str):
        """
        将提取出的特质(trait)添加到实体的 profile 中。
        如果实体 profile 字段不存在或不是列表，则进行初始化。
        """
        if not canonical_name or not trait_text:
            return

        # 确保实体在 registry 中存在
        if canonical_name not in self.registry:
            # 如果实体还不存在，先创建一个基本结构
            self.registry[canonical_name] = {
                "profile": [],  # 默认 profile 为空列表
                "aliases": [],
                "first_seen": "N/A" # 或者你可以根据实际情况传递 chunk_id
            }

        entity_data = self.registry[canonical_name]

        # 确保 'profile' 键存在且其值是一个列表
        if "profile" not in entity_data or not isinstance(entity_data["profile"], list):
            entity_data["profile"] = [] # 如果不是列表，则重新初始化为列表

        # 避免重复添加完全相同的描述
        if trait_text not in entity_data["profile"]:
            entity_data["profile"].append(trait_text)
            # === [NEW] 打印日志到控制台 ===
            print(f"   [DEBUG-REGISTRY] 📝 Added trait to '{canonical_name}': '{trait_text}'")
            # ==============================
            self._save()
        else:
            # === [NEW] 打印日志到控制台 ===
            print(f"   [DEBUG-REGISTRY] 🔄 Trait already exists for '{canonical_name}', skipping.")
            # ==============================


    def get_formatted_entities_for_prompt(self,
                                          candidate_entity_name: Optional[str] = None,
                                          top_k_global_relevant: int = 5) -> Tuple[str, str]:
        """
        Formats known entities into two sections for the LLM prompt:
        1. Recently Active Entities: from the deque.
        2. Other Globally Relevant Known Entities: (optionally) top-k by embedding similarity to a candidate.

        Returns:
            Tuple[str, str]: (formatted_recent_entities_str, formatted_other_known_entities_str)
        """
        # 1. Recently Active Entities (from deque)
        formatted_recent_entities = []
        # [FIX] 确保 recent_entities 里的名字确实在 registry 里有详细数据
        for name in self.recent_entities:
            data = self.registry.get(name)
            if data:
                aliases_str = f" (aka: {', '.join(data.get('aliases', []))})" if data.get('aliases') else ""
                profile_str = data.get('profile', 'No profile.')
                formatted_recent_entities.append(f"- {name}{aliases_str}: {profile_str}")
        recent_active_entities_str = "\n".join(formatted_recent_entities) or "None"

        # 2. Other Globally Relevant Known Entities (Top-K if embedding_model available)
        formatted_other_known_entities = []
        if self.registry:
            all_canonical_names_in_registry = list(self.registry.keys())

            # 排除掉已经在 recently active 里的实体，避免重复
            all_canonical_names_to_consider = [name for name in all_canonical_names_in_registry if
                                               name not in self.recent_entities]

            if candidate_entity_name and self.embedding_model and self.entity_embedding_store and all_canonical_names_to_consider:
                self._ensure_embeddings_for_registry()  # 确保所有 canonical names 都有嵌入

                # 嵌入 candidate_entity_name
                candidate_embedding = self.embedding_model.batch_encode(
                    [candidate_entity_name],
                    norm=True
                )[0]

                existing_canonical_entity_names = []
                existing_canonical_entity_embeddings = []

                for name in all_canonical_names_to_consider:
                    entity_hash_id = compute_mdhash_id(name, prefix="entity-")
                    if entity_hash_id in self.entity_embedding_store.hash_id_to_row:
                        embedding = self.entity_embedding_store.get_embedding(entity_hash_id)
                        if embedding is not None and len(embedding) > 0:  # 确保嵌入有效
                            existing_canonical_entity_names.append(name)
                            existing_canonical_entity_embeddings.append(embedding)

                if existing_canonical_entity_names:
                    similarities = np.dot(np.array(existing_canonical_entity_embeddings), candidate_embedding)
                    top_k_indices = np.argsort(similarities)[::-1][:top_k_global_relevant]

                    for idx in top_k_indices:
                        name = existing_canonical_entity_names[idx]
                        data = self.registry[name]
                        aliases_str = f" (aka: {', '.join(data.get('aliases', []))})" if data.get('aliases') else ""
                        profile_str = data.get('profile', 'No profile.')
                        formatted_other_known_entities.append(
                            f"- {name}{aliases_str}: {profile_str} (Similarity: {similarities[idx]:.2f})")
            else:  # Fallback: 如果没有 candidate 或嵌入模型，或者实体太少，直接列出部分
                # 避免过度，只从 registry 中取出几个，或者不使用相似度
                for i, name in enumerate(all_canonical_names_to_consider):
                    if i >= top_k_global_relevant:
                        break
                    data = self.registry[name]
                    aliases_str = f" (aka: {', '.join(data.get('aliases', []))})" if data.get('aliases') else ""
                    profile_str = data.get('profile', 'No profile.')
                    formatted_other_known_entities.append(f"- {name}{aliases_str}: {profile_str}")

        other_known_entities_str = "\n".join(formatted_other_known_entities) or "None"

        return recent_active_entities_str, other_known_entities_str

    def get_known_entities_for_prompt(self, max_tokens=1000) -> str:
        lines = []
        for name, data in self.registry.items():
            aliases = data.get("aliases", [])
            aliases_str = f" (aka: {', '.join(aliases)})" if aliases else ""

            entry = f"- {name}{aliases_str}: {data.get('profile', 'No profile.')}\n"
            lines.append(entry)
        return "".join(lines) or "No entities registered yet."
