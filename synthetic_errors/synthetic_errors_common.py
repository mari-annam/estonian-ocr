import random
import json

def add_noise(text, mappings):
    noisy_text = list(text)
    random.seed(5)

    num_replacements = max(1, int(len(text) * 0.2))

    for _ in range(num_replacements):
        positions = [i for i, char in enumerate(noisy_text) if char in dict(mappings).keys()]
        if not positions:
            break
        position = random.choice(positions)
        source_char = noisy_text[position]
        replacement_char = dict(mappings)[source_char]
        noisy_text[position] = replacement_char

    return ''.join(noisy_text)

def generate_noisy_json(input_texts, mappings):
    for entry in input_texts:
        entry["ocr_text"] = add_noise(entry["ground_truth"], mappings)
    return input_texts

mappings = [
    ('ü', 'ii'),
    ('ii', 'ü'),
    ('f', 's'),
    ('o', 'a'),
    ('a', 'o'),
    ('o', '0'),
    ('0', 'o'),
    ('i', '1'),
    ('1', 'i'),
    ('!', 'i'),
    ('i', '!'),
    ('.', ','),
    (',', '.')
]

input_file = "tasuja.json"
output_file = "synthetic_tasuja_common.json"

with open(input_file, "r", encoding="utf-8") as f:
    input_texts = json.load(f)

noisy_data = generate_noisy_json(input_texts, mappings)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(noisy_data, f, indent=4, ensure_ascii=False)