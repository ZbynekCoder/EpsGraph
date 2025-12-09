import os
import sys
import shutil
import numpy as np
from dotenv import load_dotenv

# 确保能找到项目路径
sys.path.append(os.getcwd())

from src.proprag.PropRAG import PropRAG
from src.proprag.utils.config_utils import BaseConfig
from src.proprag.graph_beam_search import BeamSearchPathFinder # 确保这是修改后的文件

load_dotenv()

def test_epistemic_search():
    # === 1. 初始化 PropRAG (复用之前的配置) ===
    my_api_key = "sk-ZHv49kkwqj8lg05MCzGYF3YFKGwZzdizk419Gv8ylT1pjhOt"
    my_base_url = "https://svip.xty.app/v1"
    my_model_name = "gpt-4.1-mini"
    output_dir = "outputs/test_graph_debug" # 复用刚才建好图的目录，不重建了！

    config = BaseConfig(
        save_dir=output_dir,
        llm_name=my_model_name,
        llm_base_url=my_base_url,
        api_key=my_api_key,
        embedding_model_name="/data-share/yeesuanAI08/zhangboyang/EpsGraph/models/NV-Embed-v2",

        # ⚠️ 关键：设为 False，直接加载刚才建好的图
        force_index_from_scratch=False,
        is_directed_graph=True
    )

    print("🚀 Loading existing Epistemic Graph...")
    rag = PropRAG(global_config=config)

    # === 2. 初始化我们的新搜索器 ===
    print("\n🔍 Initializing Epistemic Beam Searcher...")
    searcher = BeamSearchPathFinder(rag, beam_width=5, max_path_length=3)

    # === 3. 执行搜索测试 ===
    # 场景：用户问 "Did Trump win?"
    # 我们希望系统能从 "Donald Trump" 这个 Agent 出发，找到他自己的观点，以及相关的冲突观点

    query = "Did Smith meet the source?"
    agent = "The anonymous source"

    print(f"\n🧠 Query: '{query}'")
    print(f"👀 Perspective: {agent}")

    paths = searcher.find_paths(query, agent_name=agent)

    # === 4. 打印结果 ===
    print(f"\n✅ Found {len(paths)} belief paths:\n")

    for i, p in enumerate(paths):
        print(f"Path #{i+1} (Score: {p['score']:.4f}):")
        # 打印路径上的文本
        for j, text in enumerate(p['texts']):
            node_type = "Agent" if j==0 else ("Belief" if j%2!=0 else "Entity")
            indent = "  " * j
            print(f"{indent} -> [{node_type}] {text}")
        print("-" * 40)

if __name__ == "__main__":
    test_epistemic_search()
