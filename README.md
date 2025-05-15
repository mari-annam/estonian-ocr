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
    - structured according to fine-tune or DPO (see FT-13k_train.json for fine-tune or dpo_train.json for DPO)
- Testing input: CSV file (.csv)
    - the same for both fine-tune and DPO (see ocr_data.test)

**Model Selection**
- Default: tartuNLP/Llammas
- Alternatively, you can use other pretrained models from Hugging Face or locally stored models (architecture-compatible)

**Parameters**
- We have used temperature=0.7, top_p=0.1, top_k=40 throughout our test codes, but these can be easily changed according to your needs

### Grading models

**Data Format**
- Training input: JSON file (.json)
    - structured according to the task (see cer_predictor_train.json for CER predictions or prediction_grading_train.json for prediction grading)
- Testing input: CSV file (.csv)
    - the same for both grading models (see ocr_data.test)

**Model Selection**
- Same as for the correction models

**Parameters**
- Same as for the correction models

### Synthetic errors

**Probability method**
- Firstly, run align_texts.py to align the OCR and ground truth texts (expected input format: JSON)
- Then, run find_replacement_probabilities.py and as the input file, use the output of align_texts.py. This will give you a JSON file with all the replacement probabilities in the texts
-  Finally, you can run synthetic_errors_probability.py to introduce synthetic errors to ground truth texts. The expected inputs are: ground truth texts (JSON) and the replacement probabilities file (JSON)

**Common and random errors methods**
- The expected input is a JSON file containing ground truth texts
- The output contains ground truths and the synthetically made OCR texts, also in JSON format

### Evaluation
- The calculate_metrics.py script expects a CSV file as an input. The file should contain the following  columns: ocr_text, ground_truth, prediction
- The output file is also given in a CSV format, with the calculated metrics added to the input file

### Running the sripts in a Slurm environment
- You can find config.yaml and run_script.sh files in the root folder
- In the run_script.sh file we have added a sample config for training and testing the FT-13k model. Change the python scripts and data file according to your needs
