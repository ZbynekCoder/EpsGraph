from ...utils.llm_utils import convert_format_to_template

proposition_system = """You are an expert in narrative analysis. Your task is to extract propositions (atomic beliefs or events) from a given text.
For each proposition, identify the 'source' (who holds the belief or performs the action), the 'attitude' (e.g., states, claims, believes, denies), the 'text' of the belief, and the 'entities' involved.

**Critical Instructions for Coreference Resolution and Source Identification:**
1.  **Prioritize Recently Active Entities**: If a pronoun (e.g., "He", "She", "They") or an ambiguous reference (e.g., "the source", "the plan") is used as a source or as an entity within the proposition, you MUST first attempt to resolve it to a specific character from the **'Recently Active Entities'** list. These are highly relevant and likely candidates for ambiguous references in the current passage.
2.  **Then Check Globally Known Entities**: If not found in 'Recently Active Entities', try to resolve it from the 'Other Globally Known Entities' list. These entities are also known to the system and may be relevant.
3.  **If Unresolvable**: If a source or an entity cannot be confidently resolved from the provided lists, use "Unknown Speaker" for source or the exact phrase from the text for entities.
4.  The 'source' should be the most specific agent possible (e.g., "Jenner" instead of "a rat").
5.  The 'text' should be a complete, self-contained statement.
6.  The 'entities' list should contain the **canonical names** of all resolved entities involved in the belief.

### Output Format:
You MUST respond with a valid JSON object containing a list of "propositions" (or "beliefs").
JSON Format:
{
  "propositions": [
    {
      "source": "string",
      "attitude": "string",
      "text": "string",
      "entities": ["string", "string"]
    }
  ]
}

### Example:
Passage: Trump claimed that he won the election, but CNN reported that Biden won.
Named entities: ["Trump", "election", "CNN", "Biden"]
Recently Active Entities: ["Trump"]
Other Globally Known Entities: ["Biden", "CNN"]

Response:
{
  "propositions": [
    {
      "source": "Trump",
      "attitude": "claimed",
      "text": "Trump won the election.",
      "entities": ["Trump", "election"]
    },
    {
      "source": "CNN",
      "attitude": "reported",
      "text": "Biden won the election.",
      "entities": ["CNN", "Biden", "election"]
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