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

# === 配置 ===
my_api_key = "sk-ZHv49kkwqj8lg05MCzGYF3YFKGwZzdizk419Gv8ylT1pjhOt"
my_base_url = "https://svip.xty.app/v1"
my_model_name = "gpt-4.1-mini"
output_dir = "outputs/jenner_test"


def main():
    # 1. 环境清理与初始化
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    config = BaseConfig(
        save_dir=output_dir,
        llm_name=my_model_name, llm_base_url=my_base_url, api_key=my_api_key,
        embedding_model_name="/data-share/yeesuanAI08/zhangboyang/EpsGraph/models/NV-Embed-v2",
        is_directed_graph=True,
        force_index_from_scratch=False  # 手动控制
    )

    print("🚀 Initializing PropRAG for Jenner Test...")
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    rag = PropRAG(global_config=config)
    searcher = BeamSearchPathFinder(rag)
    validator = ConsistencyValidator(rag.llm_model)

    # === 2. 故事流 ===
    story_stream = [
        # Time 1: 建立人设
        "Jenner was a cynical rat who loudly opposed Nicodemus's Plan to move the colony to Thorn Valley.",

        # Time 2: 强化动机
        "He argued that stealing electricity and food from humans was easier and better than working hard in the fields.",

        # Time 3: 矛盾爆发 (OOC)
        "Jenner announced that he agreed wholeheartedly with The Plan and couldn't wait to start farming."
    ]

    print("\n🎬 --- Action! ---")

    for i, doc in enumerate(story_stream):
        print(f"\n\n📍 [Time Step {i + 1}] Input: \"{doc}\"")

        # 增量 Indexing
        rag.global_config.force_index_from_scratch = False
        rag.index([doc])

        # 获取最新的 Belief 进行审计
        # 注意：这里我们取最后一个被添加的 Belief
        # 在真实的流式场景中，我们应该监听 "New Belief Event"
        last_prop_key = list(rag.proposition_to_entities_map.keys())[-1]
        last_belief = rag.proposition_to_entities_map[last_prop_key]

        agent_name = last_belief["source"]
        new_statement = last_belief["text"]

        # 防御：如果提取出的 source 是 GlobalContext，可能不需要审计，或者审计 Narrative 一致性
        # 这里我们主要关注 Jenner
        if agent_name == "GlobalContext" and "Jenner" in new_statement:
            # 如果 LLM 把 "Jenner agreed" 提取为客观事实 (Source=Global)，
            # 我们其实应该审计的是 Jenner 这个实体。
            # 这是一个高阶技巧，暂且假设 LLM 能正确提取 Source=Jenner
            pass

        print(f"🔎 Auditing Agent: {agent_name}")
        print(f"   Statement: \"{new_statement}\"")

        # 1. 检索 Ego-centric Memories
        agent_key = compute_mdhash_id(agent_name, prefix="entity-")

        # 强制刷新缓存 (因为是同一个 searcher 对象)
        searcher._build_indexes()

        # 查找该 Agent 过去的所有 Beliefs
        # 注意：我们这里模拟的是 "自我反思"，所以检索所有关联 Belief
        # 简单起见，我们直接获取 Agent 节点直连的所有 Agency 边

        if agent_key not in searcher.agent_beliefs_cache:
            print("   (No prior memories found for this agent)")
            memories = []
        else:
            belief_keys = searcher.agent_beliefs_cache[agent_key]
            memories = []
            for bk in belief_keys:
                if bk == last_prop_key: continue  # 跳过当前这句

                b_data = rag.proposition_to_entities_map.get(bk)
                if b_data:
                    memories.append({
                        "text": b_data["text"],
                        "source": b_data["source"],
                        "nodes": [agent_key, bk],
                        "texts": [agent_name, b_data["text"]]  # 为了适配 Validator 接口
                    })

            print(f"   Found {len(memories)} prior memories.")
            for m in memories:
                print(f"   - {m['text']}")

        # 2. 调用 Validator
        result = validator.validate(
            agent_name=agent_name,
            new_belief_text=new_statement,
            retrieved_memories=memories  # Validator 内部会处理格式
        )

        print(f"🤖 Validation Result:")
        print(f"   Status: {result['status']}")
        print(f"   Reasoning: {result['reasoning']}")

        if i == 2:  # 最后一步
            if result['status'] == "Inconsistent":
                print("\n✅ SUCCESS: Inconsistency detected!")
            else:
                print("\n❌ FAILURE: Failed to detect inconsistency.")


if __name__ == "__main__":
    main()
