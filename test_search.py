import json
import os
import sys
import shutil
import numpy as np
import time # 导入 time 模块
from dotenv import load_dotenv

load_dotenv()

# 确保能找到项目路径
sys.path.append(os.getcwd())

from src.proprag.PropRAG import PropRAG
from src.proprag.utils.config_utils import BaseConfig
from src.proprag.graph_beam_search import BeamSearchPathFinder
from src.proprag.reasoning.consistency_validator import ConsistencyValidator # 导入 Validator

# === 1. 初始化 PropRAG (复用之前的配置) ===
my_api_key = "sk-ZHv49kkwqj8lg05MCzGYF3YFKGwZzdizk419Gv8ylT1pjhOt"
my_base_url = "https://svip.xty.app/v1"
my_model_name = "gpt-4.1-mini"
output_dir = "outputs/test_graph_debug"

config = BaseConfig(
    save_dir=output_dir,
    llm_name=my_model_name,
    llm_base_url=my_base_url,
    api_key=my_api_key,
    embedding_model_name="/data-share/yeesuanAI08/zhangboyang/EpsGraph/models/NV-Embed-v2", # 你的 A100 跑这个没问题
    force_index_from_scratch=False,
    is_directed_graph=True
)

print("🚀 Loading existing Epistemic Graph...")
rag = PropRAG(global_config=config)

# === 2. 初始化搜索器和验证器 ===
print("\n🔍 Initializing Epistemic Beam Searcher and Consistency Validator...")
searcher = BeamSearchPathFinder(rag, beam_width=5, max_path_length=3)
validator = ConsistencyValidator(rag.llm_model) # 传入 PropRAG 内部的 LLM 实例

# === 3. 模拟审计场景 ===
# 假设 Agent: The anonymous source, Persona: "secretive, bold, exposing corruption"

# --- Scenario 1: Consistent ---
print("\n--- Scenario 1: Consistent Statement ---")
agent_to_audit_1 = "The anonymous source"
new_statement_1 = "The anonymous source reaffirmed the deal was signed in secret, as previously reported."
agent_persona_1 = "a secretive and bold whistleblower exposing corruption"

print(f"\n🧠 Auditing Agent: '{agent_to_audit_1}'")
print(f"📝 New Statement: '{new_statement_1}'")
print(f"🎭 Persona: '{agent_persona_1}'")

# 1. Agent 检索自己的记忆 (相关信念)
related_memories_1 = searcher.find_paths(query=new_statement_1, agent_name=agent_to_audit_1)

# 2. 验证器判断一致性
start_time = time.time()
audit_result_1 = validator.validate(
    agent_name=agent_to_audit_1,
    new_belief_text=new_statement_1,
    retrieved_memories=related_memories_1,
    agent_persona=agent_persona_1
)
print(f"Audit Result (Scenario 1) took {time.time() - start_time:.2f}s:\n{json.dumps(audit_result_1, indent=2, ensure_ascii=False)}")


# --- Scenario 2: Inconsistent Statement (Hypocrisy/Change of Stance) ---
print("\n--- Scenario 2: Inconsistent Statement ---")
agent_to_audit_2 = "The anonymous source"
# 假设 Source 突然说自己没说过
new_statement_2 = "The anonymous source denied ever claiming that Governor Smith explicitly promised to pay anyone."
agent_persona_2 = "a secretive and bold whistleblower exposing corruption" # 保持人设不变

print(f"\n🧠 Auditing Agent: '{agent_to_audit_2}'")
print(f"📝 New Statement: '{new_statement_2}'")
print(f"🎭 Persona: '{agent_persona_2}'")

# 1. Agent 检索自己的记忆 (相关信念)
related_memories_2 = searcher.find_paths(query=new_statement_2, agent_name=agent_to_audit_2)

# 2. 验证器判断一致性
start_time = time.time()
audit_result_2 = validator.validate(
    agent_name=agent_to_audit_2,
    new_belief_text=new_statement_2,
    retrieved_memories=related_memories_2,
    agent_persona=agent_persona_2
)
print(f"Audit Result (Scenario 2) took {time.time() - start_time:.2f}s:\n{json.dumps(audit_result_2, indent=2, ensure_ascii=False)}")


# --- Scenario 3: Neutral/New Information (no prior conflicting memory) ---
print("\n--- Scenario 3: Neutral/New Information ---")
agent_to_audit_3 = "The anonymous source"
# 假设 Source 评论了完全无关的事件
new_statement_3 = "The anonymous source commented that the weather in Washington D.C. has been unusually warm this week."
agent_persona_3 = "a secretive and bold whistleblower exposing corruption"

print(f"\n🧠 Auditing Agent: '{agent_to_audit_3}'")
print(f"📝 New Statement: '{new_statement_3}'")
print(f"🎭 Persona: '{agent_persona_3}'")

# 1. Agent 检索自己的记忆 (相关信念)
related_memories_3 = searcher.find_paths(query=new_statement_3, agent_name=agent_to_audit_3)

# 2. 验证器判断一致性
start_time = time.time()
audit_result_3 = validator.validate(
    agent_name=agent_to_audit_3,
    new_belief_text=new_statement_3,
    retrieved_memories=related_memories_3,
    agent_persona=agent_persona_3
)
print(f"Audit Result (Scenario 3) took {time.time() - start_time:.2f}s:\n{json.dumps(audit_result_3, indent=2, ensure_ascii=False)}")


