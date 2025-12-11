import asyncio
import json
import re
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.entity_registry import GlobalEntityRegistry
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json
from ..utils.misc_utils import PropositionRawOutput, RetryExecutor

import re
import ast  # For safe evaluation of entities list structure
import json  # For loading JSON and dumping fixed structures
import logging

logger = get_logger(__name__)


@dataclass
class Proposition:
    """Class for representing a proposition extracted from text."""
    text: str
    entities: List[str]


class PropositionExtractor:
    """
    Class to extract propositions from passages before entity-relation extraction.
    Each proposition represents a fully contextualized unit of meaning.
    """

    def __init__(self, llm_model, entity_registry: Optional[GlobalEntityRegistry] = None):
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model
        self.entity_registry = entity_registry

    def extract_propositions(self, chunk_key: str, passage: str, named_entities: Optional[List[str]] = None,
                             temperature=0.0, use_cache=True) -> PropositionRawOutput:
        """
        Extract propositions from a passage.

        Args:
            chunk_key: Identifier for the chunk
            passage: The text passage to extract propositions from
            named_entities: Optional list of pre-extracted named entities to use

        Returns:
            PropositionRawOutput object containing the propositions and metadata
        """
        # 1. Prepare Prompt
        recent_active_entities_str = "None"
        other_globally_relevant_entities_str = "None"

        if self.entity_registry:
            # For proposition extraction, the 'candidate_entity_name' for Top-K relevance is the current passage itself,
            # but for coreference, we want to give *all* recently active and a broad set of global.
            # So, we pass None as candidate_entity_name to get a general list or can iterate over current named_entities.
            # For simplicity now, let's just use the general one.
            recent_active_entities_str, other_globally_relevant_entities_str = \
                self.entity_registry.get_formatted_entities_for_prompt(
                    candidate_entity_name=None,  # For proposition extraction, no specific single candidate_entity_name
                    top_k_global_relevant=10  # Get a reasonable number of global entities
                )

        # 安全检查: 确保 named_entities 是列表
        if named_entities is None:
            named_entities = []
        elif not isinstance(named_entities, list):
            logger.warning(f"named_entities expected list, got {type(named_entities)}. Resetting to empty.")
            named_entities = []

        proposition_input_message = self.prompt_template_manager.render(
            name='proposition_extraction',
            passage=passage,
            named_entities=json.dumps(named_entities),
            recent_active_entities=recent_active_entities_str,
            other_globally_known_entities=other_globally_relevant_entities_str
        )

        print(proposition_input_message)

        # 2. Extract Raw Beliefs
        try:
            raw_response, metadata, _ = self.llm_model.infer(
                messages=proposition_input_message,
                temperature=temperature,
                response_format={"type": "json_object"},
                use_cache=use_cache
            )
            print("raw_beliefs:")
            raw_beliefs = self._parse_llm_response(raw_response)

        except Exception as e:
            logger.error(f"Initial belief extraction failed for chunk {chunk_key}: {e}")
            return PropositionRawOutput(chunk_key, str(e), [], {"error": str(e)})

        # 3. Canonicalize Entities
        if not self.entity_registry:
            return PropositionRawOutput(chunk_key, raw_response, raw_beliefs, metadata)

        canonical_beliefs = []
        if self.entity_registry:
            for belief in raw_beliefs:
                source = belief.get("source", "GlobalContext")
                canonical_source = self.entity_registry.resolve_and_add(source, passage, chunk_key)

                canonical_entities = []
                for entity_name in belief.get("entities", []):
                    canonical_entity = self.entity_registry.resolve_and_add(entity_name, passage, chunk_key)
                    if canonical_entity:
                        canonical_entities.append(canonical_entity)

                belief["source"] = canonical_source
                belief["entities"] = list(set(canonical_entities))
                canonical_beliefs.append(belief)
            self.entity_registry.update_context_history(passage)
        else:
            canonical_beliefs = raw_beliefs

        return PropositionRawOutput(chunk_key, raw_response, canonical_beliefs, metadata)

        # s1 = set(named_entities)
        # s2 = set().union(*[set(prop['entities']) for prop in propositions])
        # missing_entities_count = len(s1 - s2)
        # import threading
        # # Thread-safe logging of statistics to file
        # with threading.Lock():
        #     try:
        #         with open('proposition_stats.txt', 'a') as f:
        #             f.write(f"{missing_entities_count}\n")
        #     except Exception as e:
        #         logger.warning(f"Failed to write proposition stats: {e}")

        return PropositionRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            propositions=propositions,
            metadata=metadata
        )

    def _parse_llm_response(self, response: str) -> List[Dict]:
        """
        尝试多种方法从 LLM 响应中提取 JSON，不依赖严格的正则顺序。
        """
        # 方法 1: 尝试直接解析
        try:
            data = json.loads(response)
            # [Fix] 无论是列表还是字典，只要解析成功，都交给 normalize 处理
            print(type(data))
            if isinstance(data, (list, dict)):
                return self._normalize_json_structure(data)
        except json.JSONDecodeError:
            pass

        print("Boom!!!!!!")

        # 方法 2: 提取 Markdown 代码块 ```json ... ```
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return self._normalize_json_structure(data)
            except json.JSONDecodeError:
                pass

        print("Boom!!!!!!")

        # 方法 3: 提取最外层的大括号 (处理只有一部分是 JSON 的情况)
        match = re.search(r"\{.*}", response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return self._normalize_json_structure(data)
            except json.JSONDecodeError:
                pass

        print("Boom!!!!!!")
        logger.warning(f"Failed to parse JSON from response: {response[:100]}...")
        return []

    def _normalize_json_structure(self, data: Any) -> List[Dict]:
        beliefs = []
        target_list = []

        if isinstance(data, list):
            target_list = data
        elif isinstance(data, dict):
            target_list = data.get("propositions", data.get("beliefs", [data]))

        # --- 强制检查 ---
        for item in target_list:
            # 1. 必须是字典
            assert isinstance(item, dict), f"LLM parsing error: item is not a dict. Got {type(item)}: {item}"

            # 2. 'text' 字段必须存在且为非空字符串
            text = item.get("text")
            assert text and isinstance(text, str), f"LLM parsing error: 'text' field is missing or invalid. Got: {text}"

            # 3. 'source' 字段必须存在且为非空字符串
            source = item.get("source")
            assert source and isinstance(source,
                                         str), f"LLM parsing error: 'source' field is missing or invalid. Got: {source} in belief: {item}"

            print(f"[DEBUG _normalize_json_structure] Item source: {repr(source)}, type: {type(source)}")

            # 4. 'entities' 字段必须存在且为列表
            entities = item.get("entities")
            assert isinstance(entities,
                              list), f"LLM parsing error: 'entities' field must be a list. Got: {type(entities)}"

            # 检查通过，加入结果
            beliefs.append({
                "text": text,
                "entities": entities,
                "source": source,
                "attitude": item.get("attitude", "fact")
            })

        return beliefs

    def batch_extract_propositions(self, chunks: Dict[str, Dict],
                                   named_entities_dict: Optional[Dict[str, List[str]]] = None) -> Dict[
        str, PropositionRawOutput]:
        """
        Extracts propositions from multiple chunks in parallel using asyncio.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=8) as executor:  # 限制 worker 数量，避免 LLM API 限流
            futures = {
                executor.submit(
                    self.extract_propositions,
                    chunk_key,
                    chunk_data["content"],
                    named_entities_dict.get(chunk_key) if named_entities_dict else None
                ): chunk_key
                for chunk_key, chunk_data in chunks.items()
            }
            for future in tqdm(as_completed(futures), total=len(chunks), desc="Batch Extracting Beliefs"):
                chunk_key = futures[future]
                try:
                    results[chunk_key] = future.result()
                except Exception as e:
                    logger.error(f"Chunk {chunk_key} failed in batch extraction: {e}")
                    results[chunk_key] = PropositionRawOutput(chunk_key, str(e), [], {})
        return results

    def _extract_propositions_from_response(self, response: str) -> Dict:
        """
        Extract beliefs from the LLM response.
        """
        # 清理 Markdown 标记 (Ollama 的模型经常喜欢加 ```json)
        clean_response = response.replace("```json", "").replace("```", "").strip()

        try:
            loaded_json = json.loads(clean_response)
        except json.JSONDecodeError:
            # 如果简单的解析失败，尝试原来的修复逻辑 (假定你保留了 fix_large_json_text)
            # 注意：如果 fix_large_json_text 内部逻辑太复杂且绑定了旧格式，可以直接跳过，
            # 因为 Llama-3.3-70B 生成 JSON 的能力通常很强。
            logger.warning(f"JSON parsing failed, raw: {clean_response[:50]}...")
            return {"propositions": []}

        extracted_beliefs = []

        # 优先寻找 "beliefs" 字段
        if "beliefs" in loaded_json:
            for item in loaded_json["beliefs"]:
                # 确保必需字段存在，没有的话给默认值
                belief = {
                    "text": item.get("text", ""),
                    "entities": item.get("entities", []),
                    "source": item.get("source", "GlobalContext"),
                    "attitude": item.get("attitude", "fact")
                }
                extracted_beliefs.append(belief)

        # 兼容旧的 "propositions" 字段
        elif "propositions" in loaded_json:
            for item in loaded_json["propositions"]:
                belief = {
                    "text": item.get("text", ""),
                    "entities": item.get("entities", []),
                    "source": "GlobalContext",  # 旧格式默认值
                    "attitude": "fact"
                }
                extracted_beliefs.append(belief)

        # 关键：PropRAG 的后续代码(enhanced_openie)期望返回键名为 "propositions" 的字典
        # 即使我们在逻辑上把它叫 beliefs，为了不改动更多代码，这里 key 依然叫 propositions
        return {"propositions": extracted_beliefs}

    def _extract_propositions_manually(self, response: str) -> Dict:
        """
        Manually extract propositions from text when JSON parsing fails.

        Args:
            response: The raw response text

        Returns:
            A dictionary with a "propositions" key containing extracted propositions
        """
        propositions = []

        # Look for text/entities patterns in the response
        text_pattern = r'"text":\s*"([^"]+)"'
        entities_pattern = r'"entities":\s*\[(.*?)\]'

        text_matches = re.findall(text_pattern, response.replace("'", '"'), re.DOTALL)
        entities_matches = re.findall(entities_pattern, response.replace("'", '"'), re.DOTALL)
        assert len(text_matches) == len(entities_matches)
        # If we have both text and entities, pair them up
        if text_matches and entities_matches and len(text_matches) == len(entities_matches):
            for text, entities_str in zip(text_matches, entities_matches):
                # Extract entity strings from the entities array
                entity_list = []
                for entity_match in re.finditer(r'"([^"]+)"', entities_str):
                    entity_list.append(entity_match.group(1))

                propositions.append({
                    "text": text,
                    "entities": entity_list
                })

        return {"propositions": propositions}

    def batch_extract_propositions(self, chunks: Dict[str, Dict],
                                   named_entities_dict: Optional[Dict[str, List[str]]] = None) -> Dict[
        str, PropositionRawOutput]:
        """
        Extract propositions from multiple chunks in parallel.

        Args:
            chunks: Dictionary of chunk IDs to chunk info
            named_entities_dict: Optional dictionary mapping chunk IDs to pre-extracted named entities

        Returns:
            Dictionary of chunk IDs to proposition extraction results
        """
        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        proposition_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor(max_workers=300) as executor:
            # Create proposition extraction futures for each chunk
            proposition_futures = {}
            for chunk_key, passage in chunk_passages.items():
                # Get named entities for this chunk if available
                named_entities = named_entities_dict.get(chunk_key, None) if named_entities_dict else None
                # print(f"Named entities for chunk {chunk_key}: {named_entities}")
                # Submit the task
                future = executor.submit(self.extract_propositions, chunk_key, passage, named_entities)
                proposition_futures[future] = chunk_key

            with RetryExecutor(executor, proposition_futures,
                               lambda chunk_key: ((self.extract_propositions, chunk_key, chunk_passages[chunk_key],
                                                   named_entities_dict.get(chunk_key, None)), {}),
                               desc="Extracting propositions") as retry_exec:
                def process_proposition(future, chunk_key, pbar):
                    result = future.result()
                    proposition_results_list.append(result)
                    metadata = result.metadata
                    nonlocal total_prompt_tokens, total_completion_tokens, num_cache_hit
                    total_prompt_tokens += metadata.get('prompt_tokens', 0)
                    total_completion_tokens += metadata.get('completion_tokens', 0)
                    if metadata.get('cache_hit'):
                        num_cache_hit += 1
                    pbar.set_postfix({
                        'total_prompt_tokens': total_prompt_tokens,
                        'total_completion_tokens': total_completion_tokens,
                        'num_cache_hit': num_cache_hit
                    })

                retry_exec.process(process_proposition)

        # Convert list of results to dictionary keyed by chunk ID
        proposition_results_dict = {res.chunk_id: res for res in proposition_results_list}

        return proposition_results_dict
