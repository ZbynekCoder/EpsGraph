import json
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.entity_registry import GlobalEntityRegistry
from ..utils.logging_utils import get_logger
from ..utils.misc_utils import PropositionRawOutput, RetryExecutor

logger = get_logger(__name__)


class PropositionExtractor:
    """
    Class to extract propositions and traits from passages.

    Data Flow:
    1. Input Passage -> LLM -> JSON {beliefs: [], traits: []}
    2. Beliefs -> Entity Resolution -> PropositionRawOutput.propositions
    3. Traits -> Entity Resolution -> EntityRegistry Profile Update -> PropositionRawOutput.traits
    """

    def __init__(self, llm_model, entity_registry: Optional[GlobalEntityRegistry] = None):
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model
        self.entity_registry = entity_registry

    def extract_propositions(self, chunk_key: str, passage: str, named_entities: Optional[List[str]] = None,
                             temperature=0.0, use_cache=True) -> PropositionRawOutput:
        # 1. 准备 Context
        recent, global_ents = "None", "None"
        if self.entity_registry:
            recent, global_ents = self.entity_registry.get_formatted_entities_for_prompt(top_k=10)

        # 2. 调用 LLM
        prompt = self.prompt_template_manager.render(
            name='proposition_extraction',
            passage=passage,
            named_entities=json.dumps(named_entities or []),
            recent_active_entities=recent,
            other_globally_known_entities=global_ents
        )

        raw_response, metadata, _ = self.llm_model.infer(
            messages=prompt, temperature=temperature, response_format={"type": "json_object"}, use_cache=use_cache
        )

        # 3. 解析 JSON (Fail Fast)
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError:
            logger.error(f"JSON Parse Error for {chunk_key}: {raw_response[:50]}...")
            return PropositionRawOutput(chunk_key, raw_response, [], metadata, [])

        # 4. 规范化 & 写入 Registry
        canonical_props = []
        canonical_traits = []

        if self.entity_registry:
            # 处理 Beliefs
            for item in data.get("beliefs", []):
                src = self.entity_registry.resolve_and_add(item.get("source", "GlobalContext"), passage, chunk_key)
                ents = [self.entity_registry.resolve_and_add(e, passage, chunk_key)
                        for e in (item.get("entities") if isinstance(item.get("entities"), list) else [])]

                item["source"] = src
                item["entities"] = [e for e in ents if e]  # 过滤空值
                canonical_props.append(item)

            # 处理 Traits (副作用：更新 Profile)
            for item in data.get("traits", []):
                ent = self.entity_registry.resolve_and_add(item.get("entity"), passage, chunk_key)
                trait = item.get("trait")
                if ent and trait:
                    self.entity_registry.add_profile_trait(ent, trait)
                    canonical_traits.append({"entity": ent, "trait": trait})

            # 更新历史上下文
            self.entity_registry.update_context_history(passage)
        else:
            # 无 Registry 模式
            canonical_props = data.get("beliefs", [])
            canonical_traits = data.get("traits", [])

        return PropositionRawOutput(chunk_key, raw_response, canonical_props, metadata, canonical_traits)

    def batch_extract_propositions(self, chunks: Dict[str, Dict],
                                   named_entities_dict: Optional[Dict[str, List[str]]] = None) -> Dict[str, PropositionRawOutput]:
        """
        Extracts propositions from multiple chunks in parallel.
        """
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}
        proposition_results_list = []

        # Metrics
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor(max_workers=300) as executor:  # Adjust max_workers as needed
            proposition_futures = {}
            for chunk_key, passage in chunk_passages.items():
                # Get NER results if available
                named_entities = named_entities_dict.get(chunk_key, []) if named_entities_dict else []

                future = executor.submit(self.extract_propositions, chunk_key, passage, named_entities)
                proposition_futures[future] = chunk_key

            # Using RetryExecutor to handle transient failures
            with RetryExecutor(executor, proposition_futures,
                               lambda chunk_key: ((self.extract_propositions, chunk_key, chunk_passages[chunk_key],
                                                   named_entities_dict.get(chunk_key,
                                                                           []) if named_entities_dict else []), {}),
                               desc="Extracting propositions") as retry_exec:

                def process_result(future, chunk_key, pbar):
                    try:
                        result = future.result()
                        proposition_results_list.append(result)

                        # Update metrics
                        meta = result.metadata
                        nonlocal total_prompt_tokens, total_completion_tokens, num_cache_hit
                        total_prompt_tokens += meta.get('prompt_tokens', 0)
                        total_completion_tokens += meta.get('completion_tokens', 0)
                        if meta.get('cache_hit'):
                            num_cache_hit += 1

                        pbar.set_postfix({
                            'prompt': total_prompt_tokens,
                            'compl': total_completion_tokens,
                            'cache': num_cache_hit
                        })
                    except Exception as e:
                        logger.error(f"Critical error in batch processing chunk {chunk_key}: {e}")
                        # Return empty output on failure to avoid breaking the pipeline
                        proposition_results_list.append(
                            PropositionRawOutput(chunk_key, str(e), [], {}, [])
                        )

                retry_exec.process(process_result)

        # Convert list to dict
        proposition_results_dict = {res.chunk_id: res for res in proposition_results_list}
        return proposition_results_dict
