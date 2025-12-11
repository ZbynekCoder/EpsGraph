from ...utils.llm_utils import convert_format_to_template

proposition_system = """You are an expert in narrative analysis. Your task is to extract information from text, distinguishing between **Subjective Beliefs** and **Objective Traits/Facts**.

### Output Format
You MUST respond with a JSON object. For each sentence/clause, decide its type:

1.  **If it's a Subjective Belief/Action (someone thinks, says, or does something):**
    - Use the "beliefs" list.
    - `source`: The character acting/thinking (e.g., "Jenner", "He"). PRESERVE PRONOUNS.
    - `attitude`: The verb (e.g., "argued", "thought").
    - `text`: The content of the belief.

2.  **If it's an Objective Trait/Fact (a description of a character or state):**
    - Use the "traits" list.
    - `entity`: The character being described (e.g., "Jenner").
    - `trait`: The description itself (e.g., "is a cynical rat").

**CRITICAL**: DO NOT use "Narrator" or "Unknown Source". If it's a description, it's a `trait`.

### Example 1
**Passage**: "Jenner was a cynical rat who loudly opposed Nicodemus's Plan."
**Response**:
{
  "traits": [
    {
      "entity": "Jenner",
      "trait": "is a cynical rat"
    }
  ],
  "beliefs": [
    {
      "source": "Jenner",
      "attitude": "opposed",
      "text": "Jenner opposed Nicodemus's Plan.",
      "entities": ["Jenner", "Nicodemus's Plan"]
    }
  ]
}

### Example 2
**Passage**: "He argued that stealing electricity was easier."
**Response**:
{
  "traits": [],
  "beliefs": [
    {
      "source": "He",
      "attitude": "argued",
      "text": "stealing electricity was easier.",
      "entities": ["He", "stealing electricity"]
    }
  ]
}
"""

# proposition_example_passage = """In 2020, after Apple launched the M1 chip, major software companies like Adobe optimized their applications, improving performance by up to 80% compared to Intel-based Macs."""

# proposition_example_entities = """["Apple", "M1 chip", "2020", "Adobe", "Adobe's applications", "Intel-based Macs", "80% performance improvement"]"""

# proposition_example_output = """{
#   "propositions": [
#     {
#       "text": "Apple launched the M1 chip in 2020.",
#       "entities": ["Apple", "M1 chip", "2020"]
#     },
#     {
#       "text": "Adobe optimized their applications specifically for the M1 chip after its launch.",
#       "entities": ["Adobe", "Adobe's applications", "M1 chip"]
#     },
#     {
#       "text": "Adobe's applications running on the M1 chip improved performance by up to 80% compared to the same applications running on Intel-based Macs.",
#       "entities": ["Adobe", "Adobe's applications", "M1 chip", "80% performance improvement", "Intel-based Macs"]
#     }
#   ]
# }"""

# proposition_second_example_passage = """In September 2023, Apple replaced the Lightning connector with USB-C on the iPhone 15, after the European Union passed regulations requiring all mobile devices to use a standardized charging port."""

# proposition_second_example_entities = """["Apple", "Lightning connector", "USB-C connector", "iPhone 15", "European Union", "regulations", "standardized charging port", "mobile devices", "September 2023"]"""

# proposition_second_example_output = """{
#   "propositions": [
#     {
#       "text": "The iPhone 15 uses a USB-C connector instead of the Lightning connector.",
#       "entities": ["iPhone 15", "USB-C connector", "Lightning connector"]
#     },
#     {
#       "text": "The European Union passed regulations requiring all mobile devices to use a standardized charging port.",
#       "entities": ["European Union", "regulations", "mobile devices", "standardized charging port"]
#     },
#     {
#       "text": "Apple changed from Lightning to USB-C on the iPhone 15 due to the European Union regulations in September 2023.",
#       "entities": ["Apple", "Lightning connector", "USB-C connector", "iPhone 15", "European Union", "regulations", "September 2023"]
#     }
#   ]
# }"""

draft = """
Break down the following passage into atomic propositions using ONLY the provided named entities. Make sure to:
1. Capture all explicit and implicit information
2. Be extremely specific about relationships between entities
3. Include all necessary context in each proposition
"""

proposition_frame = """
Passage:
```
{passage}
```

Pre-identified entities directly from this passage:
{named_entities}

[Contextual Information for Coreference Resolution]
Recently Active Entities (mentioned in recent passages, HIGH PRIORITY for pronouns/references):
{recent_active_entities}

Other Globally Known Entities (for broader context, ranked by relevance):
{other_globally_known_entities}

Extract all propositions from the passage, resolving all pronouns and ambiguous references to the most specific known entity from the lists above.
"""

prompt_template = [
    {"role": "system", "content": proposition_system},
    # {"role": "user", "content": proposition_frame.format(passage=proposition_example_passage, named_entities=proposition_example_entities)},
    # {"role": "assistant", "content": proposition_example_output},
    # {"role": "user", "content": proposition_frame.format(passage=proposition_second_example_passage, named_entities=proposition_second_example_entities)},
    # {"role": "assistant", "content": proposition_second_example_output},
    {"role": "user", "content": convert_format_to_template(original_string=proposition_frame, placeholder_mapping=None, static_values=None)}
]