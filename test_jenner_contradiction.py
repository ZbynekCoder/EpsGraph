import os
import shutil
import sys
from dotenv import load_dotenv
sys.path.append(os.getcwd())
from src.proprag.utils.config_utils import BaseConfig
from src.proprag.pipeline import CognitiveAuditPipeline # Import Pipeline

load_dotenv()

# Config
API_KEY = "sk-ZHv49kkwqj8lg05MCzGYF3YFKGwZzdizk419Gv8ylT1pjhOt"
BASE_URL = "https://svip.xty.app/v1"
MODEL = "gpt-4.1-mini"
EMBEDDING = "/data-share/yeesuanAI08/zhangboyang/EpsGraph/models/NV-Embed-v2"
OUTPUT_DIR = "outputs/jenner_clean_test"

def main():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)

    config = BaseConfig(
        save_dir=OUTPUT_DIR,
        llm_name=MODEL, llm_base_url=BASE_URL, api_key=API_KEY,
        embedding_model_name=EMBEDDING,
        is_directed_graph=True,
        force_index_from_scratch=True
    )

    print("🚀 Initializing Pipeline...")
    pipeline = CognitiveAuditPipeline(config)

    story = [
        "Mrs. Frisby is the widowed head of a family of field mice.",
        "Jenner was a cynical rat who loudly opposed the Plan.",
        "He argued that stealing electricity was easier than working.",
        "Jenner announced that he agreed wholeheartedly with The Plan."
    ]

    print("\n🎬 --- Start Stream ---")
    for i, text in enumerate(story):
        print(f"\n📍 [Step {i+1}] Input: \"{text}\"")

        # === One-Line Call ===
        result = pipeline.process_event(text)
        # =====================

        # Print Traits
        if result['new_traits']:
            print(f"   📝 Traits Found: {len(result['new_traits'])}")
            for t in result['new_traits']:
                print(f"      - {t['entity']}: {t['trait']}")

        # Print Beliefs & Audit
        if not result['audit_results']:
            print("   (No subjective beliefs to audit)")
        else:
            for audit in result['audit_results']:
                print(f"\n   🔎 Auditing {audit.agent}...")
                print(f"      Statement: \"{audit.statement}\"")
                print(f"      Evidence Count: {len(audit.evidence)}")
                print(f"      🤖 Status: [{audit.status}]")
                if audit.status == "Inconsistent":
                    print(f"      🚨 REASON: {audit.reasoning}")

if __name__ == "__main__":
    main()
