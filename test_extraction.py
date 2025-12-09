import os
import sys

# 确保能找到项目根目录
sys.path.append(os.getcwd())

from src.proprag.utils.config_utils import BaseConfig
from src.proprag.llm.openai_gpt import CacheOpenAI  # 既然 infer 用的是同步 client，这里直接用 OpenAI_GPT
from src.proprag.information_extraction.proposition_extraction import PropositionExtractor


def test_belief_extraction():
    # === API 配置 (请在此处填入你的真实信息) ===

    # 示例1：使用 DeepSeek API
    # api_key = "sk-xxxxxxxx"
    # base_url = "https://api.deepseek.com"
    # model_name = "deepseek-chat"

    # 示例2：使用 硅基流动 (SiliconFlow) Qwen
    # api_key = "sk-xxxxxxxx"
    # base_url = "https://api.siliconflow.cn/v1"
    # model_name = "Qwen/Qwen2.5-14B-Instruct"

    # 请在这里填入你的配置：
    my_api_key = "sk-ZHv49kkwqj8lg05MCzGYF3YFKGwZzdizk419Gv8ylT1pjhOt"
    my_base_url = "https://svip.xty.app/v1"
    my_model_name = "gpt-4.1-mini"

    print(f"🚀 Connecting to API: {my_base_url}")
    print(f"Model: {my_model_name}")

    # === 2. 初始化 Config ===
    config = BaseConfig(
        save_dir="outputs/test_debug",
        llm_name=my_model_name,
        llm_base_url=my_base_url,
        api_key=my_api_key,  # 这里必须填写真实的 Key

        embedding_model_name="none",
        temperature=0.0,
        max_new_tokens=2048
    )

    # === 3. 初始化 LLM ===
    try:
        # 使用同步的 OpenAI_GPT 类 (对应你贴出来的 infer 代码)
        llm = CacheOpenAI(cache_dir="outputs/test_debug",llm_name="gpt-4.1-mini",api_key="sk-ZHv49kkwqj8lg05MCzGYF3YFKGwZzdizk419Gv8ylT1pjhOt",llm_base_url="https://svip.xty.app/v1")
        print("✅ LLM Client Initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to init LLM: {e}")
        return

    # === 4. 初始化提取器 ===
    extractor = PropositionExtractor(llm)

    # === 5. 准备测试数据 ===
    text = (
        "The anonymous source told The Post that the deal was signed in secret. "
        "'He explicitly promised to pay us,' the source claimed, referring to Governor Smith. "
        "However, Smith's office released a statement denying any such meeting took place. "
        "'The Governor has never met this individual,' the statement read. "
        "But later that evening, a leaked memo suggested that Smith's deputy might have attended in his place."
    )

    entities = [
        "anonymous source", "The Post", "deal", "Governor Smith",
        "Smith's office", "meeting", "leaked memo", "Smith's deputy"
    ]

    print(f"\n📝 Input Text:\n{text}")
    print("\n⏳ Extracting Beliefs...")

    # === 6. 运行抽取 ===
    try:
        # 关闭 cache，确保真正请求 API
        result = extractor.extract_propositions(
            chunk_key="debug_chunk_001",
            passage=text,
            named_entities=entities,
            use_cache=False
        )

        # === 7. 打印结果 ===
        print("\n=== ✨ Extraction Result ===")

        if not result.propositions:
            print("⚠️ Result is empty.")

        for idx, belief in enumerate(result.propositions):
            source = belief.get('source', 'GlobalContext')
            attitude = belief.get('attitude', 'fact')
            content = belief.get('text', '')
            print(f"{idx + 1}. [{source}] --({attitude})--> \"{content}\"")
            print(f"   Target Entities: {belief.get('entities', [])}")

    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_belief_extraction()
