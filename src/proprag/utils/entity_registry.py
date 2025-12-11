import json
import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

import numpy as np

from src.proprag.utils.misc_utils import compute_mdhash_id

logger = logging.getLogger(__name__)


class GlobalEntityRegistry:
    def __init__(self, llm_model: Any, embedding_model: Any, entity_embedding_store: Any,
                 save_path: Optional[str] = None):
        self.registry: Dict[str, Dict[str, Any]] = {}  # Canonical Name -> {profile, aliases}
        self.reverse_lookup: Dict[str, str] = {}  # Alias -> Canonical Name
        self.recent_entities = deque(maxlen=10)
        self.context_history = deque(maxlen=3)
        self.llm = llm_model
        self.embedding_model = embedding_model
        self.entity_embedding_store = entity_embedding_store
        self.save_path = save_path

        if self.save_path and os.path.exists(self.save_path):
            self._load()
        self._ensure_embeddings_for_registry()

    def _load(self):
        with open(self.save_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.registry = data.get("registry", {})
            self.reverse_lookup = data.get("reverse_lookup", {})

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
        if text and isinstance(text, str):
            if not self.context_history or self.context_history[-1] != text:
                self.context_history.append(text)

    def add_profile_trait(self, canonical_name: str, trait_text: str):
        """
        将提取出的特质(trait)添加到实体的 profile 中。
        如果实体 profile 字段不存在或不是列表，则进行初始化。
        """
        if not canonical_name or not trait_text:
            return

        # 确保实体在 registry 中存在
        if canonical_name not in self.registry:
            # 如果实体还不存在
            self.registry[canonical_name] = {
                "profile": [],  # 默认 profile 为空列表
                "aliases": [],
                "first_seen": "N/A"  # 或者你可以根据实际情况传递 chunk_id
            }

        data = self.registry[canonical_name]
        if "profile" not in data or not isinstance(data["profile"], list):
            data["profile"] = []

        if trait_text not in data["profile"]:
            data["profile"].append(trait_text)
            logger.info(f"Updated profile for {canonical_name}: {trait_text}")
            self._save()

    def resolve_and_add(self, candidate_entity_name: str, context_text: str, chunk_id: str,
                        top_k: int = 5) -> str:
        """Synchronously resolves or adds an entity."""
        name = candidate_entity_name.strip()
        if not name: return ""

        # 1. 缓存命中
        if name in self.reverse_lookup:
            canonical = self.reverse_lookup[name]
            if canonical not in self.recent_entities: self.recent_entities.appendleft(canonical)
            return canonical

        # 2. LLM Resolution
        prev_context = "\n---\n".join(self.context_history) if self.context_history else "None"
        recent, global_ents = self.get_formatted_entities_for_prompt(name, top_k)

        prompt = f"""
                [PREVIOUS CONTEXT]:\n{prev_context}\n
                [CURRENT TEXT]: "{context_text}"\n
                [CANDIDATES]:\n{recent}\n
                [GLOBAL KNOWLEDGE]:\n{global_ents}\n

                Task: Resolve entity mention "{name}".
                Output JSON: {{"canonical_name": "...", "is_new": bool}}
                """

        try:
            raw, _, _ = self.llm.infer(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, response_format={"type": "json_object"}
            )
            data = json.loads(raw)
            canonical = data.get("canonical_name", name).strip()

            # 注册新实体
            if data.get("is_new", True) and canonical not in self.registry:
                self.registry[canonical] = {"profile": [], "aliases": [name], "first_seen": chunk_id}
            elif canonical in self.registry:
                if name not in self.registry[canonical].get("aliases", []):
                    self.registry[canonical]["aliases"].append(name)

            # 更新查找表
            self.reverse_lookup[name] = canonical
            if canonical not in self.reverse_lookup: self.reverse_lookup[canonical] = canonical

            # 更新短期记忆
            if canonical in self.recent_entities: self.recent_entities.remove(canonical)
            self.recent_entities.appendleft(canonical)

            self._save()
            return canonical

        except Exception as e:
            logger.error(f"Resolution failed for {name}: {e}")
            return name

    def get_formatted_entities_for_prompt(self,
                                          candidate_entity_name: Optional[str] = None,
                                          top_k: int = 5) -> Tuple[str, str]:
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
                    top_k_indices = np.argsort(similarities)[::-1][:top_k]

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
                    if i >= top_k:
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
