import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import importlib
from collections import defaultdict
from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import numpy as np
from collections import defaultdict
import re
import time

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from .information_extraction.enhanced_openie import EnhancedOpenIE
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .utils.entity_registry import GlobalEntityRegistry
from .utils.misc_utils import *
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple, Proposition # Triple type hint remains as per rule
from .utils.config_utils import BaseConfig
from .graph_beam_search import BeamSearchPathFinder

logger = logging.getLogger(__name__)

class PropRAG:

    def __init__(self, global_config=None, save_dir=None, llm_model_name=None, embedding_model_name=None, llm_base_url=None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific PropRAG instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (EnhancedOpenIE): The Open Information Extraction module.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            proposition_embedding_store (EmbeddingStore): The embedding store handling proposition embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed vLLM model, can be inserted directly as well as through configuration file.
        """
        if global_config is None:
            self.global_config = BaseConfig()
            print("BaseConfig!")
        else:
            self.global_config = global_config
            print("GlobalConfig!")
        print("global_config: ", self.global_config)

        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"PropRAG init with config:\n  {_print_config}\n")

        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(self.global_config)
        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity')
        self.proposition_embedding_store = EmbeddingStore(self.embedding_model,
                                                          os.path.join(self.working_dir, "proposition_embeddings"),
                                                          self.global_config.embedding_batch_size, 'proposition')

        logger.info("Using EnhancedOpenIE with proposition extraction")
        registry_path = os.path.join(self.working_dir, "entity_registry.json")

        self.entity_registry = GlobalEntityRegistry(
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
            entity_embedding_store=self.entity_embedding_store,
            save_path=registry_path)
        self.openie = EnhancedOpenIE(llm_model=self.llm_model, entity_registry=self.entity_registry)

        self.graph = self.initialize_graph()

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = os.path.join(self.global_config.save_dir,f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        self.ready_to_retrieve = False

        # [Rashomon Fix] 初始化时加载 Map
        self.load_proposition_map()
        self.node_to_node_stats = {}
        self.ent_node_to_num_chunk = {}


    def initialize_graph(self):
        """
        Initializes a graph using a GraphML file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a GraphML file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graphml_xml_file = os.path.join(
            self.working_dir, f"graph_{self.global_config.synonymy_edge_sim_threshold}.graphml"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graphml_xml_file):
                preloaded_graph = ig.Graph.Read_GraphML(self._graphml_xml_file)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def load_proposition_map(self):
        """
        [Rashomon Fix] Load the proposition_to_entities_map from disk.
        """
        map_path = os.path.join(self.working_dir, "proposition_map.json")
        if os.path.exists(map_path):
            logger.info(f"Loading proposition map from {map_path}")
            try:
                with open(map_path, 'r', encoding='utf-8') as f:
                    self.proposition_to_entities_map = json.load(f)
                logger.info(f"Loaded {len(self.proposition_to_entities_map)} propositions.")
            except Exception as e:
                logger.error(f"Failed to load proposition map: {e}")
                self.proposition_to_entities_map = {}
        else:
            logger.warning("Proposition map file not found. If this is a new index, it will be created.")
            self.proposition_to_entities_map = {}

    def index(self, docs: List[str]):
        """
        [Rashomon Fix] Incremental Indexing.
        Only processes NEW documents to avoid graph duplication.
        """
        logger.info(f"Indexing Documents (Incremental Check)...")

        # 1. 识别真正的新文档
        existing_keys = set(self.chunk_embedding_store.get_all_ids())
        self.chunk_embedding_store.insert_strings(docs)  # Store 内部会处理去重
        current_keys = set(self.chunk_embedding_store.get_all_ids())

        # 计算差集，得到本次新增的 keys
        new_chunk_keys = list(current_keys - existing_keys)

        if not new_chunk_keys:
            logger.info("No new documents detected. Skipping extraction and graph update.")
            return

        logger.info(f"Found {len(new_chunk_keys)} new documents to process.")

        # 2. 仅对新文档运行 OpenIE
        all_chunks_text = self.chunk_embedding_store.get_text_for_all_rows()
        new_rows = {k: all_chunks_text[k] for k in new_chunk_keys}

        # 批量抽取
        new_ner, _, new_props_raw = self.openie.batch_openie(new_rows)

        # 3. 更新全局 OpenIE 记录 (用于存盘备份)
        all_openie_info, _ = self.load_existing_openie(list(current_keys))
        updated_openie_info = self.merge_openie_results(
            all_openie_info, new_rows, new_ner, None, new_props_raw
        )
        self.openie_info = updated_openie_info
        if self.global_config.save_openie:
            self.save_openie_results(self.openie_info)

        # 4. 准备图更新数据
        new_propositions_flat = []
        new_chunk_prop_entities_map = []  # 对应 new_chunk_keys 的顺序

        # 辅助构建临时 Map，用于快速查找
        chunk_to_props_map = {k: v.propositions for k, v in new_props_raw.items()}

        for chunk_key in new_chunk_keys:
            props = chunk_to_props_map.get(chunk_key, [])
            # 收集该 Chunk 下所有 Prop 涉及的 Entities
            entities_in_chunk = set()
            for p in props:
                entities_in_chunk.update(p.get("entities", []))
                # 收集扁平化的 Props 用于 Embedding
                new_propositions_flat.append(p)

            new_chunk_prop_entities_map.append(list(entities_in_chunk))

        # 5. Embedding (Store 会自动处理去重，所以直接传也没事，但为了效率还是只传新的好)
        # 这里为了简单，传 new_propositions_flat，Store 内部其实是 append 模式
        if new_propositions_flat:
            logger.info(f"Encoding {len(new_propositions_flat)} new propositions...")
            self.proposition_embedding_store.insert_strings([p['text'] for p in new_propositions_flat])

        # Encoding Entities (收集所有新实体)
        all_new_entities = set()
        for ents in new_chunk_prop_entities_map:
            all_new_entities.update(ents)
        if all_new_entities:
            logger.info(f"Encoding {len(all_new_entities)} new entities...")
            self.entity_embedding_store.insert_strings(list(all_new_entities))

        # 6. 更新 Proposition Map (内存 + 磁盘)
        # [Critical Fix] 追加更新，不要清空
        for prop in new_propositions_flat:
            if "text" in prop:
                prop_key = compute_mdhash_id(prop["text"], prefix="proposition-")

                # --- 新增 DEBUG 打印 ---
                source_from_prop = prop.get("source")
                source_for_map = prop.get("source", "GlobalContext")
                print(f"[DEBUG PropRAG.index map update] prop_key: {prop_key}")
                print(
                    f"        -> source_from_prop (raw from _normalize): {repr(source_from_prop)}, type: {type(source_from_prop)}")
                print(
                    f"        -> source_for_map (with GlobalContext fallback): {repr(source_for_map)}, type: {type(source_for_map)}")
                # --- 确保这里使用的 source 变量是经过 fallback 的 ---

                self.proposition_to_entities_map[prop_key] = {
                    "text": prop["text"],
                    "entities": prop.get("entities", []),
                    "source": prop.get("source", "GlobalContext"),
                    "attitude": prop.get("attitude", "fact")
                }

        # 更新 proposition_to_passages (用于统计，可选)
        for chunk_key in new_chunk_keys:
            props = chunk_to_props_map.get(chunk_key, [])
            for p in props:
                pk = compute_mdhash_id(p["text"], prefix="proposition-")
                if not hasattr(self, 'proposition_to_passages'): self.proposition_to_passages = defaultdict(set)
                self.proposition_to_passages[pk].add(chunk_key)

        # 7. 增量建图 (直接操作 igraph)
        logger.info("Updating Graph with new elements...")
        self._add_new_proposition_edges(new_propositions_flat)
        self._add_new_passage_edges(new_chunk_keys, new_chunk_prop_entities_map)

        # 8. 同义词边 (可选：因为比较耗时且是全局的)
        # 这里先只做保存，如果需要同义词，需要谨慎处理 node_to_node_stats 防止重复
        # 简单起见，我们调用 add_synonymy_edges 但让 augment_graph 做去重检查
        self.add_synonymy_edges()
        self.augment_graph()  # 这里的 augment 现在只负责处理 node_to_node_stats 里的边

        # 保存状态
        self.save_igraph()

        map_path = os.path.join(self.working_dir, "proposition_map.json")
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(self.proposition_to_entities_map, f, ensure_ascii=False, indent=2)
        logger.info("Incremental Indexing Complete.")

    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None,
                 use_beam_search: bool = False,
                 beam_width: int = 5,
                 embedding_combination: str = "concatenate",
                 second_stage_filter_k: int = 0,
                 sim_threshold: float = 0.75) -> Union[List[QuerySolution], Tuple[List[QuerySolution], Dict]]:
        """
        Performs retrieval using the PropRAG framework, which consists of several steps:
        - Proposition Retrieval
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).
            use_beam_search: bool, optional
                If True, uses beam search for graph traversal to find relevant documents.
            beam_width: int, optional
                The number of paths to consider in beam search. Only used if use_beam_search is True.
            embedding_combination: str, optional
                Method to combine proposition embeddings in beam search paths. Options:
                - "concatenate": Re-embed the concatenated text of propositions (default)
                - "average": Use the average of embeddings
                - "weighted_average": Weight by position in path
                - "max_pool": Use element-wise maximum values
                - "attention": Weight by attention to query
                - "predictor": Use a trained model to predict combined embedding (if available)
            second_stage_filter_k: int, optional
                If > 0, apply a second stage filtering using concatenate method on top K candidates
                during beam search. Set to 0 to disable (default).
            sim_threshold: float, optional
                Threshold for considering edges as synonym connections (default: 0.75).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.
        """

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)
        retrieval_results = []
        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            sorted_doc_ids, sorted_doc_scores, paths, top_entities_and_scores, other_candidates = self.graph_search_with_proposition_entities(query=query,
                                                                                        link_top_k=self.global_config.linking_top_k,
                                                                                        passage_node_weight=self.global_config.passage_node_weight, 
                                                                                        two_stage_processing=True, 
                                                                                        focus_top_k=50)
            

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))
        if gold_docs is not None:
            k_list = [1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results], k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa(self,
               queries: List[Union[str,QuerySolution]],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None,
               use_beam_search: bool = False,
               beam_width: int = 5,
               embedding_combination: str = "concatenate",
               second_stage_filter_k: int = 0,
               sim_threshold: float = 0.75) -> Union[Tuple[List[QuerySolution], List[str], List[Dict]], Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]]:
        """
        Performs retrieval-augmented generation enhanced QA using the PropRAG framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.
            use_beam_search (bool, optional): If True, uses beam search for proposition path finding
                during retrieval.
            beam_width (int, optional): The number of paths to consider in beam search when use_beam_search
                is True. Default is 5.
            embedding_combination (str, optional): Method to combine proposition embeddings in beam search:
                - "concatenate": Re-embed the concatenated text (default)
                - "average": Use the average of embeddings
                - "weighted_average": Weight by position in path
                - "max_pool": Use element-wise maximum values
                - "attention": Weight by attention to query
                - "predictor": Use a trained model to predict combined embedding
            second_stage_filter_k (int, optional): If > 0, apply a second stage filtering with concatenate
                method on top K candidates during beam search.
            sim_threshold (float, optional): Threshold for considering edges as synonyms during beam search
                (default: 0.75).

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        logger.info(f"Starting RAG QA with {len(queries)} queries")
        
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            logger.info("Performing retrieval for raw query strings")
            if gold_docs is not None:
                logger.info("Retrieving with gold docs for evaluation")
                queries, overall_retrieval_result = self.retrieve(
                    queries=queries, 
                    gold_docs=gold_docs, 
                    use_beam_search=use_beam_search, 
                    beam_width=beam_width,
                    embedding_combination=embedding_combination,
                    second_stage_filter_k=second_stage_filter_k,
                    sim_threshold=sim_threshold
                )
            else:
                logger.info("Retrieving without gold docs")
                queries = self.retrieve(
                    queries=queries, 
                    use_beam_search=use_beam_search, 
                    beam_width=beam_width,
                    embedding_combination=embedding_combination,
                    second_stage_filter_k=second_stage_filter_k,
                    sim_threshold=sim_threshold
                )
            logger.info(f"Retrieval completed, got {len(queries)} results")

        logger.info("Performing QA on retrieved documents")
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        if gold_answers is not None:
            logger.info("Evaluating QA results against gold answers")
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            logger.info("Completed RAG QA without evaluation")
            return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List['QuerySolution']) -> Tuple[List['QuerySolution'], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model,
        leveraging multithreading for faster inference.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        all_qa_messages = []
        for query_solution in tqdm(queries, desc="Collecting QA prompts"):
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user_content = ""
            for passage in retrieved_passages:
                prompt_user_content += f'Wikipedia Title: {str(passage)}\n\n' 
            prompt_user_content += 'Question: ' + query_solution.question + '\nThought: '

            template_name = f'rag_qa_{self.global_config.dataset}'
            if not self.prompt_template_manager.is_template_name_valid(name=template_name):
                logger.debug(
                    f"{template_name} does not have a customized prompt template. Using MUSIQUE's prompt template instead."
                )
                template_name = 'rag_qa_musique' 

            rendered_message = self.prompt_template_manager.render(
                name=template_name,
                prompt_user=prompt_user_content 
            )
            all_qa_messages.append(rendered_message)

        all_qa_results = [None] * len(all_qa_messages)
        logger.info(f"Starting QA inference.")

        def response_checker(response_message, finish_reason, params):
            if finish_reason == "length":
                logger.warning(f"Response for query {response_message} finishes early due to length. Retrying with temperature += 0.1...")
                params['temperature'] += 0.1
                return False
            if 'Answer:' not in response_message:
                logger.warning(f"Response for query {response_message} does not contain 'Answer:'. Retrying with temperature += 0.1...")
                params['temperature'] += 0.1
                return False
            return True

        with ThreadPoolExecutor(max_workers=300) as executor:
            qa_futures = {}
            for i, qa_message in enumerate(all_qa_messages):
                future = executor.submit(self.llm_model.infer, qa_message, response_checker=response_checker, max_completion_tokens=2048)
                qa_futures[future] = i

            with RetryExecutor(executor, qa_futures,
                             lambda i: ((self.llm_model.infer, all_qa_messages[i]), {"response_checker": response_checker}),
                             desc="QA Reading (Parallel)") as retry_exec:
                def process_qa(future, i, pbar):
                    result = future.result()
                    all_qa_results[i] = result
                retry_exec.process(process_qa)

        logger.info("Finished QA inference.")

        if not all_qa_results:
             logger.warning("No QA results were returned from parallel inference.")
             return [], [], []

        valid_results = [res for res in all_qa_results if isinstance(res, tuple) and len(res) == 3]
        if len(valid_results) != len(all_qa_results):
            logger.warning(f"Expected 3 elements per result, but got inconsistencies. Found {len(valid_results)} valid results out of {len(all_qa_results)}.")

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results) 
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extracting Answers from LLM Response"):
            if query_solution_idx < len(all_response_message):
                response_content = all_response_message[query_solution_idx]
                pred_ans = response_content 
                if response_content: 
                    try:
                        if 'Answer:' in response_content:
                            pred_ans = response_content.split('Answer:', 1)[1].strip()
                        else:
                            logger.warning(f"Response for query {query_solution_idx} does not contain 'Answer:'. Using full response. Original response: {response_content}")
                            pred_ans = response_content 
                    except Exception as e:
                        logger.warning(f"Error parsing answer for query {query_solution_idx} from response: '{response_content}'. Error: {str(e)}")
                        pred_ans = response_content 
                else:
                     logger.warning(f"Received empty or None response for query {query_solution_idx}.")
                     pred_ans = "[No Answer Received]" 

                query_solution.answer = pred_ans
                queries_solutions.append(query_solution)
            else:
                 logger.error(f"Mismatch in number of queries and responses. Missing response for query index {query_solution_idx}.")


        return queries_solutions, all_response_message, all_metadata

    def add_passage_edges(self, chunk_ids: List[str], chunk_proposition_entities: List[List[str]]):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding proposition entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of proposition entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_proposition_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_proposition_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_text_for_all_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size,
                                                    threshold_score=self.global_config.synonymy_edge_sim_threshold)

        num_synonym_proposition = 0
        synonym_candidates = [] 

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_proposition += 1

                        self.node_to_node_stats[sim_edge] = self.node_to_node_stats.get(sim_edge, 0) + score
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             proposition_results_dict_triples: Dict[str, PropositionRawOutput], # Renamed from triple_results_dict
                             proposition_results_dict_props: Dict[str, PropositionRawOutput] = None) -> List[dict]: # Renamed from proposition_results_dict
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including propositions (if available),
        named-entity recognition (NER) entities, and (legacy) propositions/triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            proposition_results_dict_triples (Dict[str, PropositionRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE (legacy) proposition/triple extraction results.
            proposition_results_dict_props (Dict[str, PropositionRawOutput], optional): A dictionary 
                mapping chunk keys to their corresponding main proposition extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            chunk_openie_info = {
                'idx': chunk_key, 
                'passage': passage,
                'extracted_entities': ner_results_dict[chunk_key].unique_entities,
            }
            
            if proposition_results_dict_triples is not None and chunk_key in proposition_results_dict_triples:
                chunk_openie_info['extracted_triples'] = proposition_results_dict_triples[chunk_key].propositions # 'propositions' field from PropositionRawOutput
            else:
                chunk_openie_info['extracted_triples'] = [] # Retain key 'extracted_triples' for compatibility if underlying data is still triple-like
            
            if proposition_results_dict_props and chunk_key in proposition_results_dict_props:
                chunk_openie_info['propositions'] = proposition_results_dict_props[chunk_key].propositions
                
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """
        for chunk in all_openie_info:
            for i, e in enumerate(chunk['extracted_entities']):
                if not isinstance(e, str):
                    print(chunk)
                    print(e)
                chunk['extracted_entities'][i] = str(e)
        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0 and num_phrases > 0 :
            openie_dict = {'docs': all_openie_info, 'avg_ent_chars': round(sum_phrase_chars / num_phrases, 4),
                           'avg_ent_words': round(sum_phrase_words / num_phrases, 4)}
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")
        elif len(all_openie_info) > 0:
            logger.warning("No phrases extracted, cannot compute averages for OpenIE results.")
            openie_dict = {'docs': all_openie_info, 'avg_ent_chars': 0, 'avg_ent_words': 0}
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results (without averages) saved to {self.openie_results_path}")


    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """
        # Add new nodes (safety check)
        self.add_new_nodes()

        # Add edges from stats (Synonyms)
        if "name" in self.graph.vs.attribute_names():
            existing_names = set(self.graph.vs["name"])
        else:
            return

        edges_to_add = []
        weights = []

        for (u, v), w in self.node_to_node_stats.items():
            if u in existing_names and v in existing_names:
                # Check existence
                if self.graph.get_eid(u, v, error=False) == -1:
                    edges_to_add.append((u, v))
                    weights.append(w)

        if edges_to_add:
            self.graph.add_edges(edges_to_add, attributes={"weight": weights})
            logger.info(f"Augmented graph with {len(edges_to_add)} synonymy edges.")

        # Clear stats to avoid reprocessing next time
        self.node_to_node_stats = {}

    def add_new_nodes(self):
        """
        [Rashomon Modified]
        Adds new nodes to the graph from Entity, Proposition (Event), and Passage embedding stores.

        This method ensures that all three types of nodes (Agent/Entity, Belief/Event, Passage)
        are properly added to the igraph structure with their attributes.
        """
        logger.info("Adding new nodes (Agents, Beliefs, Entities, Passages) to graph...")

        # 1. 获取现有节点，避免重复
        existing_nodes = set(self.graph.vs["name"]) if "name" in self.graph.vs.attribute_names() else set()

        # 2. 从三个 Store 获取所有数据
        # 这里的 key 是 md5 hash (e.g., "entity-xxx", "proposition-yyy", "chunk-zzz")
        entity_nodes_data = self.entity_embedding_store.get_text_for_all_rows()
        passage_nodes_data = self.chunk_embedding_store.get_text_for_all_rows()
        proposition_nodes_data = self.proposition_embedding_store.get_text_for_all_rows()

        # 3. 合并所有潜在节点
        all_potential_nodes = {}
        all_potential_nodes.update(entity_nodes_data)  # type: entity
        all_potential_nodes.update(passage_nodes_data)  # type: passage
        all_potential_nodes.update(proposition_nodes_data)  # type: belief (new)

        # 4. 准备批量插入的数据结构
        # igraph.add_vertices(attributes=...) 需要一个字典，key是属性名，value是属性值列表
        attributes_batch = defaultdict(list)
        nodes_to_add_count = 0

        for node_id, node_data in all_potential_nodes.items():
            if node_id not in existing_nodes:
                nodes_to_add_count += 1

                # 必须属性: name
                attributes_batch['name'].append(node_id)

                # 内容属性: content (文本)
                attributes_batch['content'].append(node_data.get('content', ''))

                # [新增] 类型属性: type
                # 根据 ID 前缀判断
                if node_id.startswith("entity-"):
                    attributes_batch['type'].append("entity")  # Agent 也是 entity
                elif node_id.startswith("proposition-"):
                    attributes_batch['type'].append("belief")  # Event
                elif node_id.startswith("chunk-"):
                    attributes_batch['type'].append("passage")
                else:
                    attributes_batch['type'].append("unknown")

                # 其他属性 (如果有 embedding 等，也可以在这里加，只要 Store 里有)
                # 目前主要需要 name, content, type

        # 5. 批量插入
        if nodes_to_add_count > 0:
            logger.info(f"Adding {nodes_to_add_count} new vertices to the graph.")
            self.graph.add_vertices(n=nodes_to_add_count, attributes=dict(attributes_batch))
        else:
            logger.debug("No new vertices to add.")

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_graphml(self._graphml_xml_file)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        propositions, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted propositions, propositions involving passage
        nodes, synonymy propositions, and total propositions.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_propositions: The number of unique extracted propositions.
                - num_propositions_with_passage_node: The number of propositions involving at least one
                  passage node.
                - num_synonymy_propositions: The number of synonymy propositions (distinct from extracted
                  propositions and those with passage nodes).
                - num_total_propositions: The total number of propositions (edges).
        """
        graph_info = {}

        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        graph_info["num_extracted_propositions"] = len(self.proposition_embedding_store.get_all_ids())

        num_propositions_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_propositions_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_propositions_with_passage_node'] = num_propositions_with_passage_node

        graph_info['num_synonymy_propositions'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_propositions"] - num_propositions_with_passage_node

        graph_info["num_total_propositions"] = len(self.node_to_node_stats)

        return graph_info

    def _add_new_proposition_edges(self, new_propositions_flat: List[Dict]):
        """
        [Rashomon Helper] Add new Agent->Belief->Entity structures directly to graph.
        Idempotent: Checks existence before adding.
        """
        # 获取当前图的节点名集合缓存
        if "name" in self.graph.vs.attribute_names():
            existing_names = set(self.graph.vs["name"])
        else:
            existing_names = set()

        added_nodes = 0
        added_edges = 0

        for i, prop in enumerate(new_propositions_flat):
            print(f"[DEBUG] Processing Prop {i}: {prop}")

            source_raw = prop['source']
            print(f"        -> raw source: {repr(source_raw)} type: {type(source_raw)}")

            if not source_raw:
                print("        -> Source is empty/None, forcing GlobalContext")
                source_agent = "GlobalContext"
            else:
                source_agent = str(source_raw)  # 强制转字符串
            # --- DEBUG END ---

            text = prop.get("text", "")
            if not text: continue

            prop_key = compute_mdhash_id(text, prefix="proposition-")
            agent_key = compute_mdhash_id(source_agent, prefix="entity-")

            # 1. Add Agent Node
            if agent_key not in existing_names:
                self.graph.add_vertex(name=agent_key, type="entity", content=source_agent)
                existing_names.add(agent_key)
                added_nodes += 1

            # 2. Add Belief Node
            if prop_key not in existing_names:
                self.graph.add_vertex(name=prop_key, type="belief", content=text)
                existing_names.add(prop_key)
                added_nodes += 1

            # 3. Edge: Agent -> Belief (Agency)
            if self.graph.get_eid(agent_key, prop_key, error=False) == -1:
                self.graph.add_edge(agent_key, prop_key, type="agency")
                added_edges += 1

            # 4. Target Entities
            for ent_name in prop.get("entities", []):
                ent_key = compute_mdhash_id(ent_name, prefix="entity-")

                # Add Entity Node
                if ent_key not in existing_names:
                    self.graph.add_vertex(name=ent_key, type="entity", content=ent_name)
                    existing_names.add(ent_key)
                    added_nodes += 1

                # Edge: Belief -> Entity (Inclusion)
                if self.graph.get_eid(prop_key, ent_key, error=False) == -1:
                    self.graph.add_edge(prop_key, ent_key, type="inclusion")
                    added_edges += 1

        logger.info(f"Direct Graph Update: Added {added_nodes} nodes, {added_edges} edges from propositions.")

    def _add_new_passage_edges(self, chunk_keys: List[str], entities_list: List[List[str]]):
        """
        [Rashomon Helper] Add Chunk->Entity edges.
        Idempotent.
        """
        if "name" in self.graph.vs.attribute_names():
            existing_names = set(self.graph.vs["name"])
        else:
            existing_names = set()

        added_nodes = 0
        added_edges = 0

        all_chunks = self.chunk_embedding_store.get_text_for_all_rows()

        for i, chunk_key in enumerate(chunk_keys):
            # 1. Add Chunk Node
            if chunk_key not in existing_names:
                content = all_chunks.get(chunk_key, {}).get("content", "")
                self.graph.add_vertex(name=chunk_key, type="chunk", content=content)
                existing_names.add(chunk_key)
                added_nodes += 1

            # 2. Edges to Entities
            ents = entities_list[i]
            for ent_name in ents:
                ent_key = compute_mdhash_id(ent_name, prefix="entity-")
                # Entity node 应该在上面加过了，但防守一波
                if ent_key not in existing_names:
                    self.graph.add_vertex(name=ent_key, type="entity", content=ent_name)
                    existing_names.add(ent_key)

                # Edge: Chunk -> Entity (Contains)
                if self.graph.get_eid(chunk_key, ent_key, error=False) == -1:
                    self.graph.add_edge(chunk_key, ent_key, type="contains")
                    added_edges += 1

        logger.info(f"Direct Graph Update: Added {added_nodes} chunk nodes, {added_edges} chunk edges.")

    def add_proposition_edges_with_entity_connections(self):
        """
        [Rashomon Modified]
        Constructs the Agent-Belief-Entity directed graph.

        New Schema:
        1. Agent Node -> (Agency Edge) -> Event Node
        2. Event Node -> (Inclusion Edge) -> Entity Node

        This method iterates through all extracted beliefs and adds the corresponding
        nodes and edges to `self.node_to_node_stats` for batch insertion later.
        """
        # logger.info("Constructing Agent-Belief-Entity Directed Graph...")
        #
        # # 统计计数器
        # num_agency_edges = 0
        # num_inclusion_edges = 0
        #
        # # 遍历所有提取出的 Beliefs (Events)
        # # 注意：这里的 self.proposition_to_entities_map 需要在 index() 中被正确填充为 {prop_key: belief_dict}
        # # 我们稍后会在 index() 中修改填充逻辑
        #
        # for prop_key, belief_data in self.proposition_to_entities_map.items():
        #     # belief_data 现在的结构是:
        #     # {'text':..., 'entities':..., 'source':..., 'attitude':...}
        #
        #     # 1. 获取 Source Agent
        #     source_agent_name = belief_data.get('source', 'GlobalContext')
        #     attitude = belief_data.get('attitude', 'fact')
        #
        #     # 2. 生成 Agent 节点的 Key (Entity 类型的节点统一加前缀)
        #     agent_key = compute_mdhash_id(source_agent_name, prefix="entity-")
        #
        #     # 3. 生成 Event (Proposition) 节点的 Key (已经在 prop_key 中)
        #     event_key = prop_key
        #
        #     # --- 添加边: Agent -> Event (Agency) ---
        #     # 权重暂时设为 1.0，未来可以根据 attitude 的强度调整 (e.g. "doubts" = 0.5)
        #     # 我们在 node_to_node_stats 中用 tuple key 来表示边: (from, to)
        #     self.node_to_node_stats[(agent_key, event_key)] = 1.0
        #     num_agency_edges += 1
        #
        #     # --- 添加边: Event -> Entity (Inclusion) ---
        #     target_entities = belief_data.get('entities', [])
        #     for entity_name in target_entities:
        #         entity_key = compute_mdhash_id(entity_name, prefix="entity-")
        #
        #         # 避免自环 (Agent 指向包含自己的 Event)
        #         # 虽然逻辑上 Agent 确实在 Event 里，但为了图游走不回环太快，可以保留或去掉
        #         # 这里保留，因为 "Trump said Trump won" 是合理的
        #
        #         self.node_to_node_stats[(event_key, entity_key)] = 1.0
        #         num_inclusion_edges += 1
        #
        # logger.info(
        #     f"Graph Construction Stats: {num_agency_edges} Agency edges, {num_inclusion_edges} Inclusion edges.")
        pass

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'proposition': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) 
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids()) 
        self.proposition_node_keys: List = list(self.proposition_embedding_store.get_all_ids()) 

        logger.info(f"Graph has {self.graph.vcount()} vertices and {self.graph.ecount()} edges")

        igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} 
        self.node_name_to_vertex_idx = igraph_name_to_idx
        
        self.entity_node_idxs = []
        for node_key in self.entity_node_keys:
            if node_key in igraph_name_to_idx:
                self.entity_node_idxs.append(igraph_name_to_idx[node_key])
        
        self.passage_node_idxs = []
        self.passage_node_key_to_idx = {}
        for i, node_key in enumerate(self.passage_node_keys):
            if node_key in igraph_name_to_idx:
                self.passage_node_idxs.append(igraph_name_to_idx[node_key])
            self.passage_node_key_to_idx[node_key] = i
            
        
        self.proposition_node_idxs = []
        for node_key in self.proposition_node_keys:
            if node_key in igraph_name_to_idx:
                self.proposition_node_idxs.append(igraph_name_to_idx[node_key])

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))


        self.proposition_embeddings = np.array(self.proposition_embedding_store.get_embeddings(self.proposition_node_keys))

        self.prop_key_to_propositions = {prop_key: self.proposition_embeddings[i] for i, prop_key in enumerate(self.proposition_node_keys)}
        self.all_proposition_embeddings = np.array([self.prop_key_to_propositions[prop_key] for prop_key in self.proposition_embedding_store.get_all_ids()])

        self.beam_search = BeamSearchPathFinder(
            self, 
            beam_width=200,
            max_path_length=1,
            embedding_combination="average",
        )
        self.query_proposition_scores = {}

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: Union[List[str], List[QuerySolution]]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'proposition' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['proposition'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif isinstance(query, str) and (query not in self.query_to_embedding['proposition'] or query not in self.query_to_embedding['passage']):
                all_query_strings.append(query)


        if len(all_query_strings) > 0:
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_proposition = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_passage'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_proposition):
                self.query_to_embedding['proposition'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_proposition_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored 
        propositions embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with proposition embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and proposition
            embeddings. The shape of the array is determined by the number of
            propositions.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        
        query_embedding = self.query_to_embedding['proposition'].get(query, None)
        if query_embedding is None:
            logger.info("Query embedding not found in cache, generating new embedding")
            query_embedding = self.embedding_model.batch_encode([query], # batch_encode expects a list
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)[0] # Get the first (and only) embedding
        
        query_proposition_scores_val = np.dot(self.proposition_embeddings, query_embedding)
        
        if query_proposition_scores_val.ndim > 1:
            query_proposition_scores_val = np.squeeze(query_proposition_scores_val)
        
        query_proposition_scores_val = min_max_normalize(query_proposition_scores_val)
        
        return query_proposition_scores_val

    def dense_passage_retrieval(self, query: str, no_sort: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        If propositions are enabled, this function will calculate document scores
        as the maximum score of any proposition belonging to the document, rather
        than using the score of the entire passage.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        if query in self.query_proposition_scores:
            return self.query_proposition_scores[query]
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode([query], # batch_encode expects a list
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)[0] # Get the first (and only) embedding
        
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores

        query_doc_scores = min_max_normalize(query_doc_scores)

        if no_sort:
            self.query_proposition_scores[query] = (np.arange(len(query_doc_scores)), query_doc_scores)
            return self.query_proposition_scores[query]

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores


    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """        
        sorted_phrases = sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)
        top_phrases = sorted_phrases[:link_top_k]
        linking_score_map = dict(top_phrases)
        
        top_k_phrase_keys = set(linking_score_map.keys())
        entity_keys_in_graph = set()
        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrase_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0
            else:
                entity_keys_in_graph.add(phrase_key)
        
        missing_keys = top_k_phrase_keys - entity_keys_in_graph
        if missing_keys:
            logger.warning(f"{len(missing_keys)} top phrase keys not found in graph")
            logger.warning(f"Missing keys: {missing_keys}, top k phrases: {top_phrases}, entity keys in graph: {entity_keys_in_graph}")
        
        filtered_linking_score_map = {}
        for phrase_key, score in linking_score_map.items():
            if phrase_key in entity_keys_in_graph:
                filtered_linking_score_map[phrase_key] = score
        
        linking_score_map = filtered_linking_score_map
        
        if np.count_nonzero(all_phrase_weights) != len(linking_score_map):
            logger.warning(f"Weight count mismatch: {np.count_nonzero(all_phrase_weights)} weights vs {len(linking_score_map)} phrases")
            
            available_weights = np.count_nonzero(all_phrase_weights)
            if available_weights > 0:
                logger.info(f"Continuing with {available_weights} available weights")
            else:
                logger.warning("No weights available, will use empty mapping")
        
        return all_phrase_weights, linking_score_map

    def retrieve_grid_search(
        self,
        queries: List[str],
        gold_docs: List[List[str]],
        select_top_k_paths_range: List[int] = [5, 10, 15, 20, 30, 40, 50],
        select_top_k_entities_range: List[int] = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        ppr_damping_factor_range: List[float] = [0.5],
        k: int = 5,
        beam_size: int = 3
    ) -> Tuple[Dict, Dict]:
        """
        Performs grid search to optimize parameters for graph_search_with_proposition_entities_single_stage.
        
        Args:
            queries: List of query strings
            gold_docs: List of lists containing gold standard document IDs for each query
            select_top_k_paths_range: Range of values to try for select_top_k_paths
            select_top_k_entities_range: Range of values to try for select_top_k_entities
            ppr_damping_factor_range: Range of values to try for ppr_damping_factor
            k: The k value for Recall@k metric (default: 100)
            
        Returns:
            Tuple containing:
            1. Dictionary with best parameters found
            2. Dictionary with Recall@k scores for all parameter combinations
        """
        logger.info("Starting beam search grid optimization")
        logger.info(f"Parameter ranges:")
        logger.info(f"  select_top_k_paths: {select_top_k_paths_range}")
        logger.info(f"  select_top_k_entities: {select_top_k_entities_range}")
        logger.info(f"  ppr_damping_factor: {ppr_damping_factor_range}")
        logger.info(f"Beam size: {beam_size}")
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()
        self.get_query_embeddings(queries)
        retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)
        
        try:
            with open(f"grid_search_results.json") as inp:
                all_results = json.load(inp)
        except FileNotFoundError:
            all_results = {}
        
        initial_params = {
            'select_top_k_paths': 20,
            'select_top_k_entities': 40,
            'ppr_damping_factor': 0.5
        }
        best_params = initial_params.copy()
        best_recall = 0.0
        
        def evaluate_params(params: Dict) -> float:
            """Helper function to evaluate a parameter combination"""
            param_key = f"paths_{params['select_top_k_paths']}_entities_{params['select_top_k_entities']}_damping_{params['ppr_damping_factor']}"
            
            if param_key in all_results:
                return all_results[param_key][f"Recall@{k}"]
            num_to_retrieve = 200
            logger.info(f"Evaluating parameters: {params}")
            retrieval_results = []
            for query in tqdm(queries):
                doc_ids, doc_scores, _, _, _ = self.graph_search_with_proposition_entities_single_stage(
                    query=query,
                    select_top_k_paths=params['select_top_k_paths'],
                    select_top_k_entities=params['select_top_k_entities'],
                    ppr_damping_factor=params['ppr_damping_factor']
                )

                top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in doc_ids[:num_to_retrieve]]
                retrieval_results.append(QuerySolution(question=query, docs=top_k_docs, doc_scores=doc_scores[:num_to_retrieve]))

            k_list = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results], k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")
            all_results[param_key] = overall_retrieval_result
            with open(f"grid_search_results.json", "w") as f:
                json.dump(all_results, f)
            return all_results[param_key][f"Recall@{k}"]

        def get_neighbors(params: Dict, param_name: str, param_range: List) -> List[Dict]:
            """Generate neighboring parameter combinations by varying one parameter"""
            current_idx = param_range.index(params[param_name])
            right_neighbors = []
            left_neighbors = []
            
            for i in range(current_idx + 1, len(param_range)):
                neighbor = params.copy()
                neighbor[param_name] = param_range[i]
                right_neighbors.append(neighbor)
                
            for i in range(current_idx - 1, -1, -1):
                neighbor = params.copy()
                neighbor[param_name] = param_range[i]
                left_neighbors.append(neighbor)
                
            return right_neighbors, left_neighbors

        best_params = initial_params.copy()
        beam = [(best_params, evaluate_params(best_params))]
        while True:
            last_best_param = best_params.copy()
            
            for param_name, param_range in [
                ('select_top_k_paths', select_top_k_paths_range),
                ('select_top_k_entities', select_top_k_entities_range),
                ('ppr_damping_factor', ppr_damping_factor_range)
            ]:
                logger.info(f"\nOptimizing parameter: {param_name}")
                logger.info(f"Current beam:")
                for params, score in beam:
                    logger.info(f"  {params} -> Recall@{k}: {score:.4f}")
                param_range = sorted(param_range)
                
                new_beam = []
                for params, score in beam:
                    right_neighbors, left_neighbors = get_neighbors(params, param_name, param_range)
                    best_score = 0.0
                    for neighbor in right_neighbors:
                        if param_name == 'select_top_k_entities' and neighbor['select_top_k_entities'] > 4 * neighbor['select_top_k_paths']:
                            break

                        neighbor_score = evaluate_params(neighbor)
                        logger.info(f"  Neighbor: {neighbor} -> Recall@{k}: {neighbor_score:.4f}")
                        new_beam.append((neighbor, neighbor_score))
                        if neighbor_score > best_score:
                            best_score = neighbor_score
                        else:
                            break
                    best_score = 0.0
                    for neighbor in left_neighbors:
                        neighbor_score = evaluate_params(neighbor)
                        logger.info(f"  Neighbor: {neighbor} -> Recall@{k}: {neighbor_score:.4f}")
                        new_beam.append((neighbor, neighbor_score))
                        if neighbor_score > best_score:
                            best_score = neighbor_score
                        else:
                            break
                
                new_beam.extend(beam)
                new_beam.sort(key=lambda x: x[1], reverse=True)
                beam = new_beam[:beam_size]

                logger.info(f"Top {beam_size} combinations after optimizing {param_name}:")
                for params, score in beam:
                    logger.info(f"  {params} -> Recall@{k}: {score:.4f}")

            best_params, best_recall = beam[0]

            logger.info("\nCurrent grid search stage completed!")
            logger.info(f"Best parameters found: {best_params}")
            logger.info(f"Best Recall@{k}: {best_recall:.4f}")

            if best_params['select_top_k_paths'] == last_best_param['select_top_k_paths'] and best_params['select_top_k_entities'] == last_best_param['select_top_k_entities'] and best_params['ppr_damping_factor'] == last_best_param['ppr_damping_factor']:
                logger.info("No improvement, breaking")
                break
            else:
                logger.info("Improvement, continuing to next stage")
        
        return best_params, all_results

    def graph_search_with_proposition_entities_single_stage(self, query: str,
                                        select_top_k_paths = 10,
                                        select_top_k_entities = 30,
                                        ppr_damping_factor = 0.75) -> Tuple[np.ndarray, np.ndarray, List, Dict, List]:
        """
        Computes document scores based on proposition-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval NV-Embed-v2. This function combines the signal from the relevant
        propositions identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            select_top_k_paths (int): Number of top proposition paths to consider.
            select_top_k_entities (int): Number of top entities from paths to consider.
            ppr_damping_factor (float): Damping factor for Personalized PageRank.
        Returns:
            Tuple[np.ndarray, np.ndarray, List, Dict, List]: A tuple containing:
                - Document IDs sorted based on their scores.
                - PPR scores associated with the sorted document IDs.
                - Top proposition paths.
                - Top entities and their scores from paths.
                - Other candidate paths from beam search.
        """
        passage_node_weight = 0.05
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query, no_sort=True)
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        phrase_scores = {}

        passage_weights = np.where(np.isnan(passage_weights), 0, passage_weights)

        paths, other_candidates, initial_paths = self.beam_search.find_paths(query)
        paths = paths[:select_top_k_paths]

        top_entities_and_scores = self.beam_search.get_entities_from_paths(paths)[:select_top_k_entities]
        for entity_key, scores in top_entities_and_scores:
            if entity_key in self.node_name_to_vertex_idx:
                phrase_id = self.node_name_to_vertex_idx[entity_key]
                phrase_weights[phrase_id] = 1.0
                phrase_scores[entity_key] = np.sum(scores)
            else:
                logger.warning(f"Entity key {entity_key} from beam search paths not found in graph node names.")

        
        phrase_weights = min_max_normalize(phrase_weights)


        for i, doc_id in enumerate(dpr_sorted_doc_ids):
            passage_node_key = self.passage_node_keys[doc_id]
            passage_dpr_score = dpr_sorted_doc_scores[i]
            if passage_node_key in self.node_name_to_vertex_idx:
                passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
                passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            else:
                logger.warning(f"Passage node key {passage_node_key} not found in graph node names.")

        node_weights = phrase_weights + passage_weights


        try:
            assert len(top_entities_and_scores) > 0 or np.sum(node_weights) > 0, f'No phrases or significant weights found in the graph for the given query: {query}'
        except AssertionError as e:
            logger.warning(f"AssertionError: {e}")
            logger.warning(f"Query: {query}")
            logger.warning(f"Query embedding (passage): {self.query_to_embedding['passage'].get(query)}")
            logger.warning(f"DPR doc scores (sample): {dpr_sorted_doc_scores[:5]}")
            logger.warning(f"Top entities and scores: {top_entities_and_scores}")
            logger.warning(f"Paths: {paths}")
            logger.warning(f"Sum of node_weights: {sum(node_weights)}")
            # Fallback to DPR if graph signals are missing
            sorted_dpr_ids_only = np.argsort(dpr_sorted_doc_scores)[::-1]
            return sorted_dpr_ids_only, dpr_sorted_doc_scores[sorted_dpr_ids_only], paths[:5], top_entities_and_scores, other_candidates


        first_ppr_doc_ids, first_ppr_doc_scores, unsorted_first_ppr_doc_scores = self.run_ppr(node_weights, damping=ppr_damping_factor)
        
        return first_ppr_doc_ids, first_ppr_doc_scores, paths[:5], top_entities_and_scores, other_candidates

        
    def graph_search_with_proposition_entities(self, query: str,
                                        link_top_k: int,
                                        passage_node_weight: float = 0.05,
                                        two_stage_processing: bool = False,
                                        focus_top_k: int = 50) -> Tuple[np.ndarray, np.ndarray, List, Dict, List]:
        """
        Computes document scores based on proposition-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval NV-Embed-v2. This function combines the signal from the relevant
        propositions identified with passage similarity and graph-based search for enhanced result ranking.

        With focused graph approach, this runs a first iteration with the full graph and beam path length 1,
        then a second iteration with a subgraph of the top documents and beam path length 2.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            passage_node_weight (float): Default weight to scale passage scores in the graph.
            two_stage_processing (bool): Whether to use the two-iteration approach with a focused subgraph.
            focus_top_k (int): Number of top documents to include in the focused subgraph (default: 200).

        Returns:
            Tuple[np.ndarray, np.ndarray, List, Dict, List]: A tuple containing:
                - Document IDs sorted based on their scores.
                - PPR scores associated with the sorted document IDs.
                - Top proposition paths from the final stage.
                - Top entities and their scores from the final stage.
                - Other candidate paths from the final stage beam search.
        """
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query, no_sort=True)
        
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        phrase_scores_map = {} 

        passage_weights = np.where(np.isnan(passage_weights), 0, passage_weights)

        paths, other_candidates, initial_paths = self.beam_search.find_paths(query)
        paths = paths[:20] 

        top_entities_and_scores = self.beam_search.get_entities_from_paths(paths)[:40] 
        for entity_key, scores in top_entities_and_scores:
            if entity_key in self.node_name_to_vertex_idx:
                 phrase_id = self.node_name_to_vertex_idx[entity_key]
                 phrase_weights[phrase_id] = 1.0 
                 phrase_scores_map[entity_key] = np.sum(scores)
            else:
                logger.warning(f"Entity key {entity_key} from beam search paths not found in graph node names during initial weighting.")

        
        phrase_weights = min_max_normalize(phrase_weights)

        for i, doc_id in enumerate(dpr_sorted_doc_ids):
            passage_node_key = self.passage_node_keys[doc_id]
            passage_dpr_score = dpr_sorted_doc_scores[i]
            if passage_node_key in self.node_name_to_vertex_idx:
                passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
                passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            else:
                logger.warning(f"Passage node key {passage_node_key} not found in graph node names during passage weighting.")
        
        node_weights = phrase_weights + passage_weights

        try:
             assert len(top_entities_and_scores) > 0 or np.sum(node_weights) > 0, f'No phrases or significant weights found in the graph for the given query: {query}'
        except AssertionError as e:
            logger.warning(f"AssertionError during initial PPR setup: {e}")
            logger.warning(f"Query: {query}")
            # Fallback: return DPR results if graph signals are weak/absent
            sorted_dpr_ids_only = np.argsort(dpr_sorted_doc_scores)[::-1] # Ensure IDs are sorted
            return sorted_dpr_ids_only, dpr_sorted_doc_scores[sorted_dpr_ids_only], paths[:5], top_entities_and_scores, other_candidates


        first_ppr_doc_ids, first_ppr_doc_scores, unsorted_first_ppr_doc_scores = self.run_ppr(node_weights, damping=0.75)
        
        if not two_stage_processing:
            return first_ppr_doc_ids, first_ppr_doc_scores, paths[:5], top_entities_and_scores, other_candidates
        
        passage_key_to_idx = {key: idx for idx, key in enumerate(self.passage_node_keys)}
        
        top_doc_keys = [self.passage_node_keys[doc_id] for doc_id in first_ppr_doc_ids[:focus_top_k]]
        
        # Ensure all top_doc_keys are in the graph before proceeding
        valid_top_doc_keys = [key for key in top_doc_keys if key in self.node_name_to_vertex_idx]
        if len(valid_top_doc_keys) < len(top_doc_keys):
            logger.warning(f"Only {len(valid_top_doc_keys)} out of {len(top_doc_keys)} top doc keys found in graph for focused search.")
        if not valid_top_doc_keys:
            logger.warning("No valid top documents for focused search, returning first iteration results.")
            return first_ppr_doc_ids, first_ppr_doc_scores, paths[:5], top_entities_and_scores, other_candidates

        top_propositions = flatten_propositions([self.chunk_propositions[doc_key] for doc_key in valid_top_doc_keys if doc_key in self.chunk_propositions])
        top_proposition_keys = [compute_mdhash_id(prop['text'], prefix="proposition-") for prop in top_propositions]
        
        top_doc_vertices = [self.node_name_to_vertex_idx[key] for key in valid_top_doc_keys]
        
        vertices_to_include = set(top_doc_vertices)
        
        entity_vertices = set()
        for doc_vertex in top_doc_vertices:
            neighbors = self.graph.neighbors(doc_vertex, mode="all")
            for neighbor in neighbors:
                if self.graph.vs[neighbor]["name"].startswith("entity-"):
                    entity_vertices.add(neighbor)
                    vertices_to_include.add(neighbor)
        
        if not vertices_to_include:
             logger.warning("No vertices to include in subgraph for focused search. Returning first iteration results.")
             return first_ppr_doc_ids, first_ppr_doc_scores, paths[:5], top_entities_and_scores, other_candidates

        subgraph = self.graph.induced_subgraph(list(vertices_to_include))
        
        if subgraph.vcount() == 0:
            logger.warning("Subgraph is empty for focused search. Returning first iteration results.")
            return first_ppr_doc_ids, first_ppr_doc_scores, paths[:5], top_entities_and_scores, other_candidates

        subgraph_node_name_to_vertex_idx = {node["name"]: idx for idx, node in enumerate(subgraph.vs)}

        orig_to_sub = {self.node_name_to_vertex_idx[name]: i_sub for name, i_sub in subgraph_node_name_to_vertex_idx.items() if name in self.node_name_to_vertex_idx}
        
        sub_to_name = {i: v for v, i in subgraph_node_name_to_vertex_idx.items()}
        
        original_graph = self.beam_search.active_graph
        original_beam_width = self.beam_search.beam_width
        original_path_length = self.beam_search.max_path_length
        original_node_name_to_vertex_idx = self.beam_search.node_name_to_vertex_idx
        original_second_stage_filter_k = self.beam_search.second_stage_filter_k
        
        self.beam_search.active_graph = subgraph
        # self.beam_search.beam_width = 4
        # self.beam_search.max_path_length = 3 
        # self.beam_search.second_stage_filter_k=40

        self.beam_search.beam_width=self.global_config.beam_width
        self.beam_search.max_path_length=self.global_config.max_path_length
        self.beam_search.second_stage_filter_k=self.global_config.second_stage_filter_k
        self.beam_search.set_node_name_to_vertex_idx(subgraph_node_name_to_vertex_idx)
        self.beam_search.clear_caches()
        
        final_focused_paths, final_other_candidates, final_initial_paths = [], [], []
        final_focused_entities_and_scores = []

        try:
            try:
                focused_paths_iter2, other_candidates_iter2, initial_paths_iter2 = self.beam_search.find_paths(query, prop_set=top_proposition_keys)
                final_focused_paths = focused_paths_iter2[:5]
                final_other_candidates = other_candidates_iter2
                final_initial_paths = initial_paths_iter2

            except KeyError as e:
                logger.warning(f"Error finding paths in focused subgraph: {str(e)}. Using first iteration results.")
                return first_ppr_doc_ids, first_ppr_doc_scores, paths[:5], top_entities_and_scores, other_candidates
            
            focused_entities_and_scores_iter2 = self.beam_search.get_entities_from_paths(final_focused_paths)[:5]
            final_focused_entities_and_scores = focused_entities_and_scores_iter2
            initial_entities_and_scores_iter2 = self.beam_search.get_entities_from_paths(final_initial_paths)[:5]

            sub_phrase_weights = np.zeros(subgraph.vcount())
            initial_sub_phrase_weights = np.zeros(subgraph.vcount())
            sub_passage_weights = np.zeros(subgraph.vcount())
            
            for entity_key, scores in focused_entities_and_scores_iter2:
                if entity_key in self.node_name_to_vertex_idx:
                    orig_vertex = self.node_name_to_vertex_idx[entity_key]
                    if orig_vertex in orig_to_sub:
                        sub_vertex = orig_to_sub[orig_vertex]
                        max_score = np.max(scores) if scores else 0.0
                        sub_phrase_weights[sub_vertex] = max_score
                    else:
                        logger.warning(f"Entity key {entity_key} (orig_vertex {orig_vertex}) not in orig_to_sub map for focused subgraph weighting.")
                else:
                    logger.warning(f"Entity key {entity_key} from focused paths not in main graph node names.")


            for entity_key, scores in initial_entities_and_scores_iter2:
                if entity_key in self.node_name_to_vertex_idx:
                    orig_vertex = self.node_name_to_vertex_idx[entity_key]
                    if orig_vertex in orig_to_sub:
                        sub_vertex = orig_to_sub[orig_vertex]
                        max_score = np.max(scores) if scores else 0.0
                        initial_sub_phrase_weights[sub_vertex] = max_score
                    else:
                         logger.warning(f"Entity key {entity_key} (initial, orig_vertex {orig_vertex}) not in orig_to_sub map.")
                else:
                    logger.warning(f"Entity key {entity_key} (initial) from focused paths not in main graph node names.")


            
            sub_phrase_weights = min_max_normalize(sub_phrase_weights) if np.sum(sub_phrase_weights) > 0 else sub_phrase_weights
            initial_sub_phrase_weights = min_max_normalize(initial_sub_phrase_weights) if np.sum(initial_sub_phrase_weights) > 0 else initial_sub_phrase_weights
            
            sub_phrase_weights = np.max([sub_phrase_weights, initial_sub_phrase_weights], axis=0)
            
            for i, doc_id in enumerate(dpr_sorted_doc_ids):
                passage_key = self.passage_node_keys[doc_id]
                if passage_key in self.node_name_to_vertex_idx:
                    orig_vertex = self.node_name_to_vertex_idx[passage_key]
                    if orig_vertex in orig_to_sub:
                        sub_vertex = orig_to_sub[orig_vertex]
                        sub_passage_weights[sub_vertex] = dpr_sorted_doc_scores[i] * passage_node_weight
                else:
                    logger.warning(f"Passage key {passage_key} from DPR not in main graph node names during focused subgraph weighting.")


            sub_node_weights = sub_phrase_weights + sub_passage_weights
            
            if np.sum(sub_node_weights) == 0:
                logger.warning("All node weights in subgraph are zero. PPR might not be effective. Returning first iteration results.")
                return first_ppr_doc_ids, first_ppr_doc_scores, paths[:5], top_entities_and_scores, other_candidates

            reset_prob_sub = np.where(np.isnan(sub_node_weights) | (sub_node_weights < 0), 0, sub_node_weights)
            sub_pagerank_scores = subgraph.personalized_pagerank(
                vertices=range(subgraph.vcount()),
                damping=0.45,
                directed=False,
                weights='weight',
                reset=reset_prob_sub,
                implementation='prpack'
            )
            final_doc_scores = unsorted_first_ppr_doc_scores.copy()

            lowest_focused_score = min(sub_pagerank_scores) if sub_pagerank_scores else 0
            non_focused_scaling = lowest_focused_score * 0.5
            final_doc_scores = final_doc_scores * non_focused_scaling
            
            for sub_idx, score in enumerate(sub_pagerank_scores):
                if sub_idx < len(sub_to_name): # Check if sub_idx is a valid key
                    name = sub_to_name[sub_idx]
                    if name.startswith("chunk-") and name in passage_key_to_idx:
                        doc_idx = passage_key_to_idx[name]
                        final_doc_scores[doc_idx] = score
                else:
                    logger.warning(f"Subgraph index {sub_idx} out of bounds for sub_to_name map (size {len(sub_to_name)}).")

            
            final_doc_ids = np.argsort(final_doc_scores)[::-1]
            final_doc_scores_sorted = final_doc_scores[final_doc_ids]
        finally:
            self.beam_search.active_graph = original_graph
            self.beam_search.beam_width = original_beam_width
            self.beam_search.max_path_length = original_path_length
            self.beam_search.second_stage_filter_k = original_second_stage_filter_k
            self.beam_search.set_node_name_to_vertex_idx(original_node_name_to_vertex_idx)
            self.beam_search.clear_caches()
        
        assert len(final_doc_ids) == len(self.passage_node_idxs), \
            f"Doc prob length {len(final_doc_ids)} != corpus length {len(self.passage_node_idxs)}"
        
        return final_doc_ids, final_doc_scores_sorted, final_focused_paths, final_focused_entities_and_scores, final_other_candidates
    
    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float =0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order. The third array contains the unsorted relevance scores
                for each document passage, matching the order of `self.passage_node_idxs`.
        """
        if damping is None: damping = 0.5 
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        
        if np.sum(reset_prob) == 0:
            logger.warning("PPR reset probabilities sum to zero. PageRank scores may be uniform or zero.")
            # Fallback: create uniform scores or zero scores if PPR won't run meaningfully
            num_passage_nodes = len(self.passage_node_idxs)
            doc_scores = np.zeros(num_passage_nodes)
            if num_passage_nodes > 0 :
                doc_scores = np.ones(num_passage_nodes) / num_passage_nodes
            
            sorted_doc_ids = np.argsort(doc_scores)[::-1] # Will be arbitrary but consistent
            sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]
            return sorted_doc_ids, sorted_doc_scores, doc_scores

        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores, doc_scores