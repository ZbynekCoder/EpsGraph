import os
import sys
import shutil
from collections import Counter

# 确保能找到项目路径
sys.path.append(os.getcwd())

from src.proprag.PropRAG import PropRAG
from src.proprag.utils.config_utils import BaseConfig
from dotenv import load_dotenv

load_dotenv()

def test_graph_construction():
    # === 1. 配置参数 ===
    # 使用你刚才跑通 Extraction 的同一套 API 配置
    my_api_key = "sk-ZHv49kkwqj8lg05MCzGYF3YFKGwZzdizk419Gv8ylT1pjhOt"
    my_base_url = "https://svip.xty.app/v1"
    my_model_name = "gpt-4.1-mini"

    # 输出目录 (每次清空以便重新建图)
    output_dir = "outputs/test_graph_debug"
    if os.path.exists(output_dir):
        print(f"Cleaning up old output dir: {output_dir}")
        shutil.rmtree(output_dir)

    print(f"🚀 Initializing PropRAG with API: {my_base_url}")

    # === 2. 初始化 Config ===
    config = BaseConfig(
        save_dir=output_dir,

        # LLM 配置
        llm_name=my_model_name,
        llm_base_url=my_base_url,
        api_key=my_api_key,

        # Embedding 配置 (既然你有 A100，这里用原版的 NV-Embed 或者你喜欢的模型)
        # 如果你想快一点，可以用轻量级的 embedding 模型，或者就用默认的
        embedding_model_name="/data-share/yeesuanAI08/zhangboyang/EpsGraph/models/NV-Embed-v2",
        # embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",  # 用个小的跑得快

        # 关键图配置
        force_index_from_scratch=True,  # 强制重建图
        is_directed_graph=True,  # ⚠️ 必须为 True，这对应我们修改后的有向逻辑

        # 其他
        max_new_tokens=2048,
        temperature=0.0
    )

    # === 3. 初始化 PropRAG ===
    # PropRAG 内部会自动初始化 LLM 和 EmbeddingModel
    rag = PropRAG(global_config=config)

    # === 4. 准备数据 ===
    # 还是那个经典的罗生门例子
    docs = [
        "The anonymous source told The Post that the deal was signed in secret. "
        "'He explicitly promised to pay us,' the source claimed, referring to Governor Smith. "
        "However, Smith's office released a statement denying any such meeting took place. "
        "'The Governor has never met this individual,' the statement read. "
        "But later that evening, a leaked memo suggested that Smith's deputy might have attended in his place."
    ]

    print("\n📝 Input Document:")
    print(docs[0])

    # === 5. 运行 Index (抽取 + Embedding + 建图) ===
    print("\n⏳ Running Indexing (Extraction + Graph Construction)...")
    # 这一步会调用我们修改过的:
    # 1. openie.batch_openie (用到新 Prompt)
    # 2. add_proposition_edges_with_entity_connections (用到新建图逻辑)
    # 3. add_new_nodes (用到新节点类型逻辑)
    rag.index(docs)

    # === 6. 验证图结构 ===
    g = rag.graph
    print(f"\n✅ Graph Constructed Successfully!")
    print(f"Nodes: {g.vcount()}")
    print(f"Edges: {g.ecount()}")

    # 6.1 检查节点类型分布
    # 我们在 add_new_nodes 里加了 type 属性，现在检查一下
    if "type" in g.vs.attribute_names():
        types = g.vs["type"]
        type_counts = Counter(types)
        print(f"\n📊 Node Type Distribution: {dict(type_counts)}")

        # 验证是否有 belief 类型的节点
        if "belief" in type_counts or "proposition" in type_counts:
            # 注意：如果你的 add_new_nodes 逻辑里是用 'proposition-' 前缀判断并赋值为 'belief'
            print("   -> 成功检测到 Belief/Event 节点！")
        else:
            print("   ⚠️ 警告：未检测到 'belief' 类型节点，请检查 add_new_nodes 逻辑。")
    else:
        print("   ⚠️ 警告：图中没有 'type' 属性，请检查 add_new_nodes 代码。")

    # 6.2 检查边 (Source -> Target)
    print("\n🕸️ Sample Edges (Source -> Target):")
    nodes = g.vs

    # 打印前 30 条边，看看连接关系
    edge_count = 0
    for edge in g.es:
        source_idx = edge.source
        target_idx = edge.target

        source_node = nodes[source_idx]
        target_node = nodes[target_idx]

        s_type = source_node["type"] if "type" in source_node.attribute_names() else "unknown"
        t_type = target_node["type"] if "type" in target_node.attribute_names() else "unknown"

        s_name = source_node["name"]
        t_name = target_node["name"]

        # 我们只关心 entity/belief 相关的边，忽略同义词边等干扰
        if s_type in ["entity", "belief"] and t_type in ["entity", "belief"]:
            print(f"[{s_type}] {s_name[:25]}... --> [{t_type}] {t_name[:25]}...")
            edge_count += 1
            if edge_count >= 20: break


if __name__ == "__main__":
    test_graph_construction()
