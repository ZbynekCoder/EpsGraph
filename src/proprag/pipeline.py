import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from .PropRAG import PropRAG
from .graph_beam_search import BeamSearchPathFinder
from .reasoning.consistency_validator import ConsistencyValidator
from .utils.misc_utils import compute_mdhash_id

logger = logging.getLogger(__name__)


@dataclass
class AuditResult:
    agent: str
    statement: str
    status: str
    reasoning: str
    evidence: List[str]


class CognitiveAuditPipeline:
    def __init__(self, config):
        self.rag = PropRAG(global_config=config)
        self.searcher = BeamSearchPathFinder(self.rag)
        self.validator = ConsistencyValidator(self.rag.llm_model)
        self.searcher._build_indexes()  # Init cache

    def process_event(self, text: str) -> Dict[str, Any]:
        """
        处理单条流式事件。
        Returns: {
            "new_beliefs": [],
            "new_traits": [],
            "audit_results": [AuditResult]
        }
        """
        # 1. Snapshot
        keys_before = set(self.rag.proposition_to_entities_map.keys())

        # 2. Index
        self.rag.global_config.force_index_from_scratch = False
        self.rag.index([text])

        # 3. Diff Beliefs
        keys_after = set(self.rag.proposition_to_entities_map.keys())
        new_keys = list(keys_after - keys_before)

        # 4. Extract Traits (Debug info retrieval)
        new_traits = []
        chunk_key = self.rag.chunk_embedding_store.text_to_hash_id.get(text)
        if hasattr(self.rag, 'openie_info') and isinstance(self.rag.openie_info, list):
            for item in self.rag.openie_info:
                if item.get('idx') == chunk_key:
                    new_traits = item.get('traits', [])
                    break

        # 5. Audit Loop
        self.searcher._build_indexes()  # Refresh graph cache
        audit_results = []

        for prop_key in new_keys:
            belief = self.rag.proposition_to_entities_map[prop_key]
            agent = belief['source']
            statement = belief['text']

            if agent == "GlobalContext": continue  # Skip objective facts

            # A. Get Profile
            profile_list = self.rag.entity_registry.registry.get(agent, {}).get("profile", [])
            profile_str = "; ".join(profile_list) if isinstance(profile_list, list) else str(profile_list)

            # B. Get Memories (Excluding current batch)
            memories = self._get_memories(agent, exclude_keys=set(new_keys))

            # C. Validate
            res = self.validator.validate(agent, statement, memories, profile_str)

            audit_results.append(AuditResult(
                agent=agent,
                statement=statement,
                status=res['status'],
                reasoning=res['reasoning'],
                evidence=[m['text'] for m in memories]
            ))

        return {
            "new_beliefs": [self.rag.proposition_to_entities_map[k] for k in new_keys],
            "new_traits": new_traits,
            "audit_results": audit_results
        }

    def _get_memories(self, agent_name: str, exclude_keys: set) -> List[Dict]:
        agent_key = compute_mdhash_id(agent_name, prefix="entity-")
        memories = []
        if agent_key in self.searcher.agent_beliefs_cache:
            for bk in self.searcher.agent_beliefs_cache[agent_key]:
                if bk in exclude_keys: continue
                b_data = self.rag.proposition_to_entities_map.get(bk)
                if b_data:
                    memories.append({
                        "text": b_data["text"],
                        "source": b_data["source"],
                        "nodes": [agent_key, bk],  # Mock path
                        "texts": [agent_name, b_data["text"]]
                    })
        return memories
