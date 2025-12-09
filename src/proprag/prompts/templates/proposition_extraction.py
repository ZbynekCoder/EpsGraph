from ...utils.llm_utils import convert_format_to_template

proposition_system = """Your task is to analyze text passages and extract 'Belief Triples' to construct a cognitive graph.

A **Belief Triple** represents a specific agent's subjective view. It consists of:
1. **source**: The entity holding the belief (e.g., "Tom", "The report"). If it is an objective fact, use "GlobalContext".
2. **attitude**: The verb connecting the agent to the claim (e.g., "believes", "claims", "denies", "reported").
3. **text**: The content of the belief as a standalone statement.
4. **entities**: A list of specific entities involved in the proposition.

**CRITICAL: Reference Resolution**: You MUST replace pronouns (he, she, it) and ambiguous references (this individual, the meeting) with specific names in the "text" field.
  - BAD: "He met him."
  - GOOD: "Governor Smith met the anonymous source."

### Output Format:
You MUST respond with a valid JSON object containing a list of "beliefs".
JSON Format:
{
  "beliefs": [
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

Response:
{
  "beliefs": [
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

Named entities: {named_entities}"""

prompt_template = [
    {"role": "system", "content": proposition_system},
    # {"role": "user", "content": proposition_frame.format(passage=proposition_example_passage, named_entities=proposition_example_entities)},
    # {"role": "assistant", "content": proposition_example_output},
    # {"role": "user", "content": proposition_frame.format(passage=proposition_second_example_passage, named_entities=proposition_second_example_entities)},
    # {"role": "assistant", "content": proposition_second_example_output},
    {"role": "user", "content": convert_format_to_template(original_string=proposition_frame, placeholder_mapping=None, static_values=None)}
]