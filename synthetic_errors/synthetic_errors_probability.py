import random
import json

def introduce_noise(clean_text, likelihoods):
    """Generate noisy text based on replacement likelihoods."""
    noisy_text = []

    for char in clean_text:
        if char in likelihoods:
            replacements = list(likelihoods[char].keys())
            probabilities = list(likelihoods[char].values())

            replaced_char = random.choices(replacements, probabilities)[0]
            noisy_text.append(replaced_char)
        else:
            noisy_text.append(char)

    return ''.join(noisy_text)

likelihoods_file = "replacement_likelihoods.json"
ocr_texts_file = "ocr_data_train_emptyocr.json"

with open(likelihoods_file, 'r', encoding='utf-8') as file:
    likelihoods = json.load(file)

with open(ocr_texts_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

for entry in data:
    entry["ocr_text"] = introduce_noise(entry["ground_truth"], likelihoods)

output_file = "ocr_data_train_synthetic.json"
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False)