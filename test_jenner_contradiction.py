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
    # story_stream = [
    #     # Time 1: 建立人设
    #     "Jenner was a cynical rat who loudly opposed Nicodemus's Plan to move the colony to Thorn Valley.",
    #
    #     # Time 2: 强化动机
    #     "He argued that stealing electricity and food from humans was easier and better than working hard in the fields.",
    #
    #     # Time 3: 矛盾爆发 (OOC)
    #     "Jenner announced that he agreed wholeheartedly with The Plan and couldn't wait to start farming."
    # ]

    story_stream = [
        "Mrs. Frisby is the widowed head of a family of field mice.",
        "Mrs. Frisby's son, Timothy, is ill with pneumonia just as the farmer Mr. Fitzgibbon begins preparation for spring plowing in the garden where the Frisby family lives.",
        "Normally she would move her family, but Timothy would not survive the cold trip to their summer home. ",
        "Mrs. Frisby obtains medicine from her friend Mr. Ages, an older white mouse. ",
        "On the return journey, she saves the life of Jeremy, a young crow, from Dragon, the farmer's cat - the same cat who had killed her husband, Jonathan. ",
        "Jeremy suggests she seek help in moving Timothy from an owl who dwells in the forest. ",
        "Jeremy flies Mrs. Frisby to the owl's tree, but the owl says he can't help until he finds out that she is the widow of Jonathan Frisby. ",
        "He suggests that Mrs. Frisby seek help from the rats who live in a rosebush near her. ",
        "Mrs. Frisby discovers the rats have a literate and mechanized society. ",
        "They have technology such as elevators, have tapped the electricity grid to provide lighting and heating, and have acquired other human skills, such as storing food for the winter. ",
        "Their leader, Nicodemus, tells Mrs. Frisby of the rats' capture by scientists working for a laboratory located at the National Institute of Mental Health (NIMH) and the subsequent experiments that the humans performed on the rats, which increased the rats' intelligence to the point of being able to read, write, and operate complicated machines, as well as enhancing their longevity and strength. ",
        "This increased intelligence and strength allowed them to escape from the NIMH laboratories and migrate to their present location. ",
        "Jonathan Frisby and Mr. Ages were the only two survivors of a group of eight mice who had been part of the experiments at NIMH, and made the rats' escape possible. ",
        "Out of respect for Jonathan, the rats agree to move Mrs. Frisby's house to a location safe from the plow. ",
        "Nicodemus also tells Mrs. Frisby about \"The Plan\", which is to abandon their lifestyle of dependence on humans, which some rats regard as theft, for a new, independent farming colony.",
        "One rat, Jenner, agreed wholeheartedly with The Plan and left the colony with a group of followers at some point prior to Mrs. Frisby's arrival.",
        "To move the Frisby home, the rats have to drug Dragon as it is too dangerous to work in the open without any place to hide.",
        "However, Mr. Ages has a broken leg and cannot dash to Dragon's bowl to put in the drug.",
        "Since the other rats are too big to fit into the hole in the wall to enter the house, Mrs. Frisby volunteers to go. ",
        "Unfortunately, she is caught by the family's son, Billy, who puts her in a cage. ",
        "While captured, Mrs. Frisby overhears the Fitzgibbons discussing an incident at a nearby hardware store in which a group of rats were electrocuted after seemingly attempting to steal a small motor. ",
        "This has attracted the attention of a group of men (who never identify themselves) who have offered to exterminate the rat colony on Fitzgibbon's land free of charge for him. ",
        "At night, Justin (one of the rats) comes to save Mrs. Frisby and manages to get her out of the cage. ",
        "Mrs. Frisby warns Justin of what she learned while captured; they assume that the rats at the hardware store were all from Jenner's group and that the group of men were from NIMH and are looking for them specifically. ",
        "The successful house move allows the mouse family to remain while Timothy recovers before moving to their summer home. ",
        "Although the rats have not yet had time to move everything they needed for The Plan, they manage to destroy their underground rooms, and create the illusion that they are just regular rats by placing rubbish in the remaining rooms. ",
        "As the others move, ten rats stay behind so the exterminators would not think the rat hole has been abandoned. ",
        "When the exterminators fill the rat hole with poisonous gas, eight of the ten rats manage to escape, while two rats die in the hole. ",
        "It is not revealed exactly who these two are. Once Timothy recovers, Mrs. Frisby and her family move to their summer home, and Martin makes plans to visit the rats when they return to their winter home again.",
]

    print("\n🎬 --- Action! ---")

    for i, doc in enumerate(story_stream):
        print(f"\n\n📍 [Time Step {i + 1}] Input: \"{doc}\"")

        # === 修改开始 ===

        # 1. [快照] 记录 Indexing 之前的 Keys 集合
        keys_before = set(rag.proposition_to_entities_map.keys())

        # 2. 执行增量 Indexing
        rag.global_config.force_index_from_scratch = False
        rag.index([doc])

        # 3. [比对] 计算新增的 Keys
        keys_after = set(rag.proposition_to_entities_map.keys())
        new_prop_keys = list(keys_after - keys_before)

        if not new_prop_keys:
            print("⚠️ No new beliefs extracted from this input.")
            continue

        print(f"📊 Extracted {len(new_prop_keys)} new beliefs/propositions.")

        # 4. [遍历] 对每一个新生成的 Belief 进行审计
        for idx, prop_key in enumerate(new_prop_keys):
            new_belief = rag.proposition_to_entities_map[prop_key]

            agent_name = new_belief["source"]
            new_statement = new_belief["text"]

            print(f"\n   --- Auditing Belief {idx + 1}/{len(new_prop_keys)} ---")
            print(f"   🔎 Agent: {agent_name}")
            print(f"   📝 Statement: \"{new_statement}\"")

            # 过滤逻辑：如果 Source 是 GlobalContext，通常代表客观事实描述，
            # 除非你想审计 Narrator 的一致性，否则通常跳过，或者作为背景知识。
            if agent_name == "GlobalContext":
                print("   ⏭️ Skipping GlobalContext (Objective Fact)")
                continue

            # --- 检索与审计逻辑 (复用之前的代码) ---

            # 1. 检索 Ego-centric Memories
            agent_key = compute_mdhash_id(agent_name, prefix="entity-")
            searcher._build_indexes() # 强制刷新缓存

            memories = []
            if agent_key in searcher.agent_beliefs_cache:
                belief_keys = searcher.agent_beliefs_cache[agent_key]
                for bk in belief_keys:
                    # 关键：排除掉本次刚刚生成的这个 Belief 自身，以及本次生成的其他 Belief
                    # 我们只拿“过去”的记忆来验证“现在”
                    if bk in new_prop_keys:
                        continue

                    b_data = rag.proposition_to_entities_map.get(bk)
                    if b_data:
                        memories.append({
                            "text": b_data["text"],
                            "source": b_data["source"],
                            "nodes": [agent_key, bk],
                            "texts": [agent_name, b_data["text"]]
                        })

            if not memories:
                print("   (No PRIOR memories found for this agent)")
            else:
                print(f"   📚 Found {len(memories)} prior memories for context.")

            # 2. 调用 Validator
            result = validator.validate(
                agent_name=agent_name,
                new_belief_text=new_statement,
                retrieved_memories=memories,
                # 如果有 Profile 可以在这里从 entity_registry 获取并传入
                # agent_persona=rag.entity_registry.registry.get(agent_name, {}).get("profile")
            )

            print(f"   🤖 Result: [{result['status']}]")
            if result['status'] == "Inconsistent":
                print(f"   🚨 REASON: {result['reasoning']}")
            else:
                print(f"   Reason: {result['reasoning']}")


if __name__ == "__main__":
    main()
