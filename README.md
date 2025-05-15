# Estonian OCR Post-Processing

Ajalooliste eestikeelsete OCR tekstide järeltöötluse ja hindamise automatiseerimine Eesti Rahvusraamatukogu jaoks (2025, TalTech)

Authors: 
- Mari-Anna Meimer
- Loore Lehtmets

Hugging Face:
- https://huggingface.co/mariannam

This repository contains the core resources developed for our Bachelor's thesis at Tallinn University of Technology. The thesis aims to support the National Library of Estonia in correcting and evaluating digitized historical texts.

Included in this repository are:
- Training and testing scripts for our models (OCR correction, Character Error Rate estimation, and prediction grading)
    - Correction: finetune_train.py, finetune_test.py, pref_alignment_train.py, pref_alignment_test.py
    - CER estimation: cer_predictor_train.py, cer_predictor_test.py
    - Prediction grading: prediction_grading_train.py, prediction_grading_test.py
- Training and testing datasets
- Scripts for injecting synthetic errors to ground truths
    - align_texts.py => find_replacement_probabilities.py => synthetic_errors_probability.py
    - synthetic_errors_random.py
    - synthetic_errors_common.py
- Evaluation tools for model performance
    - calculate_metrics.py
- Tools to assess how the models perform when used together

For error correction, we explored two approaches: fine-tuning and Direct Preference Optimization (DPO). However, the models published on Hugging Face were trained using fine-tuning only. For evaluation, we've tried both fine-tuning and a Reward Model, but again, only the fine-tuned versions are available on Hugging Face. This is purely due to the fact that the fine-tuned models performed much better in both cases.

## Using the models

### Correction models

**Data Format**
- Training input: JSON file (.json)
    - Format: structured according to fine-tune or DPO (see FT-13k_train.json for fine-tune or dpo_train.json for DPO)
- Testing input: CSV file (.csv)
    - Format: the same for both fine-tune and DPO (see ocr_data.test)

**Model Selection**
- Default: tartuNLP/Llammas
- Alternatively, you can use other pretrained models from Hugging Face or locally stored models (architecture-compatible)



