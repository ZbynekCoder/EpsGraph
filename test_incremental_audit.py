import os
import shutil
import sys

from dotenv import load_dotenv

sys.path.append(os.getcwd())

from src.proprag.PropRAG import PropRAG
from src.proprag.utils.config_utils import BaseConfig
from src.proprag.graph_beam_search import BeamSearchPathFinder
from src.proprag.reasoning.consistency_validator import ConsistencyValidator
from src.proprag.utils.misc_utils import compute_mdhash_id

load_dotenv()

my_api_key = "sk-ZHv49kkwqj8lg05MCzGYF3YFKGwZzdizk419Gv8ylT1pjhOt"
my_base_url = "https://svip.xty.app/v1"
my_model_name = "gpt-4.1-mini"
output_dir = "outputs/incremental_audit_run"

def main():
    # 1. 初始清理一次
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # 2. 初始化 Config (注意：force_index_from_scratch 只在最开始为 True)
    config = BaseConfig(
        save_dir=output_dir,
        llm_name=my_model_name, llm_base_url=my_base_url, api_key=my_api_key,
        embedding_model_name="/data-share/yeesuanAI08/zhangboyang/EpsGraph/models/NV-Embed-v2",
        is_directed_graph=True,
        # 初始设为 False，我们在代码里手动控制重建逻辑
        force_index_from_scratch=False
    )

    print("Initializing Memory Manager (PropRAG)...")
    # 第一次运行，确保没有残留数据，手动清理
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    rag = PropRAG(global_config=config)

    story_chunks = [
        "The anonymous source told The Post that the deal was signed in secret.",
        "'He explicitly promised to pay us,' the source claimed, referring to Governor Smith.",
        "However, Smith's office released a statement denying any such meeting took place.",
    ]

    print("\n--- Starting Incremental Audit Process ---")

    for i, doc in enumerate(story_chunks):
        print(f"\n\n--- [Time Step {i + 1}] New Information Arrives ---")
        print(f"Text: \"{doc}\"")

        # === 关键修改 ===
        # 每次调用 index，确保追加而不是覆盖
        # PropRAG.index() 默认会追加到现有的 EmbeddingStore
        # 但是！如果每次都 new PropRAG()，且 force_from_scratch=True，就会清空。
        # 我们这里复用了 rag 对象，所以内存里的图谱一直是累积的。

        # 唯一需要注意的是：PropRAG.index() 内部是否会强制清空？
        # 检查你的 PropRAG.index() 代码。如果是原版，它会根据 config.force_index_from_scratch 来决定。
        # 我们需要在循环里，动态修改 config (或者传入参数)

        rag.global_config.force_index_from_scratch = False  # 强制设为 False，保证追加
        rag.index([doc])
        print(f"DEBUG: Registry Keys: {list(rag.entity_registry.registry.keys())}")
        print(f"DEBUG: Reverse Lookup 'source' -> {rag.entity_registry.reverse_lookup.get('source', 'Not Found')}")

        # ... (后续的 Audit 逻辑不变) ...
        # 获取最新的 Belief，审计它

        last_belief_key = list(rag.proposition_to_entities_map.keys())[-1]
        last_belief_data = rag.proposition_to_entities_map[last_belief_key]
        agent_to_audit = last_belief_data["source"]
        new_statement = last_belief_data["text"]

        print(f"  -> Auditing Agent '{agent_to_audit}'")

        # C. Perform Audit
        searcher = BeamSearchPathFinder(rag)
        validator = ConsistencyValidator(rag.llm_model)

        # 1. 获取 Agent 的所有直接信念 (从内存中的图谱里拿)
        agent_key_to_audit = compute_mdhash_id(agent_to_audit, prefix="entity-")
        all_agent_belief_keys = searcher.agent_beliefs_cache.get(agent_key_to_audit, [])

        formatted_memories = []
        for b_key in all_agent_belief_keys:
            if b_key == last_belief_key: continue  # 排除现在这句

            # 这里拿到的，就是【截止到目前为止】Agent 积累的所有旧记忆
            belief_data = rag.proposition_to_entities_map.get(b_key)
            if belief_data:
                formatted_memories.append({
                    "nodes": [agent_key_to_audit, b_key],
                    "texts": [agent_to_audit, belief_data["text"]],
                    "score": 1.0,
                    "source": agent_to_audit
                })

        # 2. 运行验证
        audit_result = validator.validate(
            agent_name=agent_to_audit,
            new_belief_text=new_statement,
            retrieved_memories=formatted_memories
        )

        print("  -> Audit Result:")
        print(f"     Status: {audit_result['status']}")
        print(f"     Reasoning: {audit_result['reasoning']}")


if __name__ == "__main__":
    main()