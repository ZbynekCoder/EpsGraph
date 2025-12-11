"""
Beam search implementation for PropRAG's entity-based knowledge graph
[Rashomon Modified] - Strict Ego-centric Search
"""

import os
import logging
import numpy as np
from typing import List, Dict, Optional, Any
import torch

from .utils.misc_utils import compute_mdhash_id

logger = logging.getLogger(__name__)

class BeamSearchPathFinder:
    """
    Epistemic Beam Search for Rashomon.

    STRICT MODE:
    Starts from a specific Agent node and finds paths ONLY within that Agent's
    subjective belief subgraph. It prohibits jumping to other agents' private beliefs
    via shared entities, enforcing epistemic privacy.
    """

    def __init__(self, prop_rag, beam_width: int = 4, max_path_length: int = 3,
                 sim_threshold: float = 0.75, **kwargs):
        self.rag = prop_rag
        self.beam_width = beam_width
        self.max_path_length = max_path_length
        self.sim_threshold = sim_threshold

        # 缓存
        self.agent_beliefs_cache = {}   # agent -> [belief_keys] (Agency)
        self.belief_entities_cache = {} # belief -> [entity_keys] (Inclusion)
        self.entity_beliefs_cache = {}  # entity -> [belief_keys] (Reverse Inclusion)

        # 预加载图结构
        self._build_indexes()

    def _build_indexes(self):
        """
        Builds fast lookup maps from the graph.
        """
        g = self.rag.graph
        nodes = g.vs

        self.agent_beliefs_cache.clear()
        self.belief_entities_cache.clear()
        self.entity_beliefs_cache.clear()

        # 简单的 Debug 开关，避免刷屏
        debug_mode = logger.isEnabledFor(logging.DEBUG)

        if debug_mode:
            print(f"[DEBUG] Graph has {len(g.es)} edges. Start indexing...")

        for edge in g.es:
            source_idx = edge.source
            target_idx = edge.target
            source_name = nodes[source_idx]["name"]
            target_name = nodes[target_idx]["name"]

            # 识别节点类型
            s_type = "belief" if source_name.startswith("proposition-") else "entity"
            t_type = "belief" if target_name.startswith("proposition-") else "entity"

            # Index 1: Agent -> Belief (Agency)
            if s_type == "entity" and t_type == "belief":
                if source_name not in self.agent_beliefs_cache:
                    self.agent_beliefs_cache[source_name] = []
                self.agent_beliefs_cache[source_name].append(target_name)

            # Index 2: Belief -> Entity (Inclusion)
            if s_type == "belief" and t_type == "entity":
                if source_name not in self.belief_entities_cache:
                    self.belief_entities_cache[source_name] = []
                self.belief_entities_cache[source_name].append(target_name)

                # Index 3: Entity -> Belief (Reverse Inclusion for traversal)
                if target_name not in self.entity_beliefs_cache:
                    self.entity_beliefs_cache[target_name] = []
                self.entity_beliefs_cache[target_name].append(source_name)

        if debug_mode:
            print(f"[DEBUG] Agent Cache Size: {len(self.agent_beliefs_cache)}")

    def get_proposition_text(self, prop_key: str) -> str:
        prop_data = self.rag.proposition_to_entities_map.get(prop_key)
        if prop_data and isinstance(prop_data, dict):
            return prop_data.get("text", "")
        return self.rag.proposition_embedding_store.get_row(prop_key).get("content", "")

    def get_belief_owner(self, prop_key: str) -> str:
        """Helper to get the source agent of a belief."""
        prop_data = self.rag.proposition_to_entities_map.get(prop_key)
        if prop_data and isinstance(prop_data, dict):
            return prop_data.get("source", "Unknown")
        return "Unknown"

    def get_proposition_embedding(self, prop_key: str) -> np.ndarray:
        emb = self.rag.proposition_embedding_store.get_embedding(prop_key)
        if isinstance(emb, torch.Tensor):
            return emb.cpu().numpy()
        return emb

    def find_paths(self, query: str, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Finds belief paths relevant to the query, strictly constrained by the agent's perspective.
        """
        logger.info(f"Starting Epistemic Search for Agent: {agent_name} | Query: {query}")

        # 1. 编码 Query
        query_embedding = self.rag.embedding_model.batch_encode(query, norm=True)
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()

        beam = []

        # 2. 确定起点 (Initial Beam) - 必须从 Agent 自己的 Beliefs 开始
        if agent_name:
            agent_key = compute_mdhash_id(agent_name, prefix="entity-")

            if agent_key not in self.agent_beliefs_cache:
                logger.warning(f"Agent {agent_name} ({agent_key}) not found or has no beliefs.")
                return []
            else:
                initial_beliefs = self.agent_beliefs_cache[agent_key]

                for belief_key in initial_beliefs:
                    belief_emb = self.get_proposition_embedding(belief_key)
                    if belief_emb is None: continue
                    score = np.dot(belief_emb, query_embedding.T).item()

                    beam.append({
                        "path": [agent_key, belief_key],
                        "score": score,
                        "history_set": {agent_key, belief_key}
                    })
        else:
            # 如果没指定 Agent，暂不支持（或者可以实现上帝视角的全局搜索）
            logger.warning("Global search not implemented in Strict Mode.")
            return []

        # 排序并截断初始 Beam
        beam.sort(key=lambda x: x["score"], reverse=True)
        beam = beam[:self.beam_width]

        # 初始化结果集：直接相关的信念也是合法路径
        final_paths = [p for p in beam]

        # 3. 扩展 Beam (Multi-hop)
        for depth in range(self.max_path_length):
            new_beam = []

            for path_obj in beam:
                current_path = path_obj["path"]
                last_node = current_path[-1]

                # Case A: Belief -> Entity (往下钻取实体)
                if last_node.startswith("proposition-"):
                    entities = self.belief_entities_cache.get(last_node, [])
                    for entity_key in entities:
                        if entity_key in path_obj["history_set"]: continue

                        new_path = current_path + [entity_key]
                        new_beam.append({
                            "path": new_path,
                            "score": path_obj["score"],
                            "history_set": path_obj["history_set"] | {entity_key}
                        })

                # Case B: Entity -> Belief (联想其他信念)
                # === [Rashomon Critical Logic] ===
                elif last_node.startswith("entity-"):
                    related_beliefs = self.entity_beliefs_cache.get(last_node, [])

                    for next_belief in related_beliefs:
                        if next_belief in path_obj["history_set"]: continue

                        # --- 严格的主观性检查 (Strict Subjectivity Check) ---
                        owner = self.get_belief_owner(next_belief)

                        # 规则：Agent 只能联想到【自己】持有的信念，或者【公理】。
                        # 绝对禁止通过实体 Entity 跳到【别的 Agent】的私有信念上！
                        if agent_name and owner != agent_name and owner != "GlobalContext":
                            # 跳过！哪怕这个信念跟 Query 再相关，Agent 也不知道（除非 Router 告诉他）
                            continue
                        # --------------------------------------------------

                        # 计算分数
                        belief_emb = self.get_proposition_embedding(next_belief)
                        if belief_emb is None: continue
                        sim = np.dot(belief_emb, query_embedding.T).item()

                        new_score = max(path_obj["score"], sim)
                        new_path = current_path + [next_belief]

                        new_beam.append({
                            "path": new_path,
                            "score": new_score,
                            "history_set": path_obj["history_set"] | {next_belief}
                        })

            if not new_beam:
                break

            new_beam.sort(key=lambda x: x["score"], reverse=True)
            beam = new_beam[:self.beam_width]

            for p in beam:
                if p["path"][-1].startswith("proposition-"):
                    final_paths.append(p)

        # 4. 格式化输出
        final_paths.sort(key=lambda x: x["score"], reverse=True)
        return self._format_paths(final_paths[:self.beam_width])

    def _format_paths(self, paths):
        formatted = []
        for p in paths:
            path_nodes = p["path"]
            texts = []
            for node in path_nodes:
                if node.startswith("entity-"):
                    texts.append(self.get_entity_text(node))
                else:
                    texts.append(self.get_proposition_text(node))

            formatted.append({
                "nodes": path_nodes,
                "texts": texts,
                "score": p["score"],
                "source": self.get_belief_owner(path_nodes[-1]) if path_nodes[-1].startswith("proposition-") else None
            })
        return formatted

    def get_entity_text(self, entity_key):
        row = self.rag.entity_embedding_store.get_row(entity_key)
        return row.get("content", entity_key)
