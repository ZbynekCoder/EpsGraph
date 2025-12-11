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
        """
        Extract propositions (beliefs) and traits from a passage.
        """
        # --- 1. Prompt Construction ---
        recent_active_entities_str = "None"
        other_globally_relevant_entities_str = "None"

        if self.entity_registry:
            recent_active_entities_str, other_globally_relevant_entities_str = \
                self.entity_registry.get_formatted_entities_for_prompt(
                    candidate_entity_name=None,
                    top_k_global_relevant=10
                )

        if named_entities is None:
            named_entities = []

        # Rendering the prompt
        # Ensure your PromptTemplateManager uses the NEW logic (Subjective/Objective split)
        proposition_input_message = self.prompt_template_manager.render(
            name='proposition_extraction',
            passage=passage,
            named_entities=json.dumps(named_entities),
            recent_active_entities=recent_active_entities_str,
            other_globally_known_entities=other_globally_relevant_entities_str
        )

        # --- 2. LLM Inference ---
        # No try-except here. If LLM API fails, let it crash so we know.
        raw_response, metadata, _ = self.llm_model.infer(
            messages=proposition_input_message,
            temperature=temperature,
            response_format={"type": "json_object"},
            use_cache=use_cache
        )

        # --- 3. Parsing & Cleaning ---
        try:
            parsed_json = json.loads(raw_response)
        except json.JSONDecodeError:
            logger.error(f"JSON Decode Error for chunk {chunk_key}. Response: {raw_response[:100]}...")
            # Return empty structure on parse failure, or raise error if you prefer strictness
            return PropositionRawOutput(chunk_key, raw_response, [], {}, [], metadata)

        # Defensive access: ensure lists
        raw_beliefs = parsed_json.get("beliefs", [])
        if not isinstance(raw_beliefs, list): raw_beliefs = []

        raw_traits = parsed_json.get("traits", [])
        if not isinstance(raw_traits, list): raw_traits = []

        # --- 4. Logic Branching: With vs Without Registry ---

        canonical_propositions = [] # Maps to 'beliefs'
        canonical_traits = []       # Maps to 'traits'

        if self.entity_registry:
            # === Process Beliefs ===
            for belief in raw_beliefs:
                # 4.1. Clean Data: Ensure entities is a list
                ents = belief.get("entities")
                if ents is None:
                    ents = []
                elif not isinstance(ents, list):
                    # Handle case where LLM outputs single string instead of list
                    ents = [str(ents)]

                # 4.2. Resolve Source (e.g., "He" -> "Jenner")
                source = belief.get("source", "GlobalContext")
                canonical_source = self.entity_registry.resolve_and_add(source, passage, chunk_key)

                # 4.3. Resolve Entities
                canonical_entities = []
                for entity_name in ents:
                    canonical_entity = self.entity_registry.resolve_and_add(entity_name, passage, chunk_key)
                    if canonical_entity:
                        canonical_entities.append(canonical_entity)

                # 4.4. Reconstruct Belief
                belief["source"] = canonical_source
                belief["entities"] = list(set(canonical_entities))
                canonical_propositions.append(belief)

            # === Process Traits ===
            for trait_item in raw_traits:
                entity_name = trait_item.get("entity")
                trait_text = trait_item.get("trait")

                if entity_name and trait_text:
                    # Resolve Entity
                    canonical_entity = self.entity_registry.resolve_and_add(entity_name, passage, chunk_key)

                    # Update Registry Profile (Side Effect)
                    self.entity_registry.add_profile_trait(canonical_entity, trait_text)

                    # Add to output list
                    canonical_traits.append({
                        "entity": canonical_entity,
                        "trait": trait_text
                    })

            # === Update History Context ===
            # This is crucial for the "Previous Context" logic in EntityRegistry
            self.entity_registry.update_context_history(passage)

        else:
            # No Registry: Pass through raw data but clean 'entities' field
            for belief in raw_beliefs:
                if belief.get("entities") is None:
                    belief["entities"] = []
                canonical_propositions.append(belief)
            canonical_traits = raw_traits

        # --- 5. Return ---
        return PropositionRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            propositions=canonical_propositions, # 'beliefs' go here
            metadata=metadata,
            traits=canonical_traits              # 'traits' go here
        )

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

        with ThreadPoolExecutor(max_workers=300) as executor: # Adjust max_workers as needed
            proposition_futures = {}
            for chunk_key, passage in chunk_passages.items():

                # Get NER results if available
                named_entities = named_entities_dict.get(chunk_key, []) if named_entities_dict else []

                future = executor.submit(self.extract_propositions, chunk_key, passage, named_entities)
                proposition_futures[future] = chunk_key

            # Using RetryExecutor to handle transient failures
            with RetryExecutor(executor, proposition_futures,
                               lambda chunk_key: ((self.extract_propositions, chunk_key, chunk_passages[chunk_key],
                                                   named_entities_dict.get(chunk_key, []) if named_entities_dict else []), {}),
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
