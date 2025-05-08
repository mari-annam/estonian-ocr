# Based on a pseudocode from an article, called "An Unsupervised method for OCR Post-Correction and Spelling Normalisation for Finnish"

import random
import json

def add_noise(text):
"""Generate noisy text based on random replacements."""
    actions = ['add', 'delete', 'replace']
    noisy_text = list(text)
    random.seed(5)

    num_errors = max(1, int(len(text) * 0.15))

    for _ in range(num_errors):
        action = random.choice(actions)
        if action == 'add':
            random_char = random.choice("abcdefghijklmnopqrstuvwäåö")
            position = random.randint(0, len(noisy_text))
            noisy_text.insert(position, random_char)
        elif action == 'delete' and len(noisy_text) > 0:
            position = random.randint(0, len(noisy_text) - 1)
            noisy_text.pop(position)
        elif action == 'replace' and len(noisy_text) > 0:
            position = random.randint(0, len(noisy_text) - 1)
            noisy_text[position] = random.choice("abcdefghijklmnopqrstuvwäåö")

    return ''.join(noisy_text)

def generate_noisy_json(input_texts):
    for entry in input_texts:
        entry["ocr_text"] = add_noise(entry["ground_truth"])
    return input_texts

input_file = "tasuja.json"
output_file = "synthetic_tasuja_random.json"

with open(input_file, "r", encoding="utf-8") as f:
    input_texts = json.load(f)

noisy_data = generate_noisy_json(input_texts)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(noisy_data, f, indent=4, ensure_ascii=False)