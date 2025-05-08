import json
import os
import random
from collections import defaultdict

likelihoods_file = "replacement_likelihoods.json"
aligned_texts = "aligned_texts.json"

def find_probability(aligned_texts):
    """Compute and save character replacement likelihoods from aligned texts."""

    with open(likelihoods_file, 'r', encoding='utf-8') as file:
        return json.load(file)

    with open(aligned_texts, 'r', encoding='utf-8') as file:
        data = json.load(file)

    replacement_counts = defaultdict(lambda: defaultdict(int))
    gt_char_totals = defaultdict(int)

    for entry in data:
        aligned_ocr_text = entry["aligned_ocr_text"]
        aligned_gt_text = entry["aligned_gt_text"]

        for ocr_char, gt_char in zip(aligned_ocr_text, aligned_gt_text):
            replacement_counts[gt_char][ocr_char] += 1
            gt_char_totals[gt_char] += 1

    replacement_likelihoods = {
        gt_char: {ocr_char: count / gt_char_totals[gt_char]
                  for ocr_char, count in ocr_char_map.items()}
        for gt_char, ocr_char_map in replacement_counts.items()
    }

    with open(likelihoods_file, 'w', encoding='utf-8') as file:
        json.dump(replacement_likelihoods, file, indent=4, ensure_ascii=False)

    return replacement_likelihoods

likelihoods = find_probability(aligned_texts)