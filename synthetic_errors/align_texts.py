import json
from Bio.Align import PairwiseAligner

def align_texts(ocr_text, gt_text):
    """
    Align texts using PairwiseAligner.
    """
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -1

    alignment = aligner.align(ocr_text, gt_text)[0]

    aligned_ocr_text = []
    aligned_gt_text = []

    ocr_index, gt_index = 0, 0
    for ocr_range, gt_range in zip(alignment.aligned[0], alignment.aligned[1]):
        while ocr_index < ocr_range[0]:
            aligned_ocr_text.append(ocr_text[ocr_index])
            aligned_gt_text.append("-")
            ocr_index += 1
        while gt_index < gt_range[0]:
            aligned_ocr_text.append("-")
            aligned_gt_text.append(gt_text[gt_index])
            gt_index += 1

        while ocr_index < ocr_range[1] and gt_index < gt_range[1]:
            aligned_ocr_text.append(ocr_text[ocr_index])
            aligned_gt_text.append(gt_text[gt_index])
            ocr_index += 1
            gt_index += 1

    while ocr_index < len(ocr_text):
        aligned_ocr_text.append(ocr_text[ocr_index])
        aligned_gt_text.append("-")
        ocr_index += 1
    while gt_index < len(gt_text):
        aligned_ocr_text.append("-")
        aligned_gt_text.append(gt_text[gt_index])
        gt_index += 1

    return "".join(aligned_ocr_text), "".join(aligned_gt_text)

def process_json(json_file, output_file):
    """
    Process a JSON file to align OCR and GT texts at the character level.
    """
    with open(json_file, 'r', encoding="utf-8") as file:
        data = json.load(file)

    results = []
    for entry in data:
        ocr_text = entry.get("ocr_text", "")
        gt_text = entry.get("ground_truth", "")

        aligned_ocr, aligned_gt = align_texts(ocr_text, gt_text)

        results.append({
            "original_ocr_text": ocr_text,
            "original_gt_text": gt_text,
            "aligned_ocr_text": aligned_ocr,
            "aligned_gt_text": aligned_gt
        })

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=4, ensure_ascii=False)


input_json = "ocr_data.json"
output_json = "aligned_texts.json"
process_json(input_json, output_json)