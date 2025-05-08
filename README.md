# Ajalooliste eestikeelsete OCR tekstide järeltöötluse ja hindamise automatiseerimine Eesti Rahvusraamatukogu jaoks (2025, TalTech)
# Automation of Post-Processing and Evaluation of Historical Estonian OCR Texts for the National Library of Estonia (2025, TalTech)

Authors: 
- Mari-Anna Meimer
- Loore Lehtmets

Hugging Face:
- https://huggingface.co/mariannam

This repository contains the core resources developed for our Bachelor's thesis at Tallinn University of Technology. The thesis aims to support the National Library of Estonia in correcting and evaluating digitized historical texts.

Included in this repository are:
- Training and testing scripts for our models (OCR correction, Character Error Rate estimation, and prediction grading)
- Training and testing datasets
- Scripts for injecting synthetic errors to ground truths
- Evaluation tools for model performance
- Tools to assess how the models perform when used together

For error correction, we explored two approaches: fine-tuning and Direct Preference Optimization (DPO). However, the models published on Hugging Face were trained using fine-tuning only. For evaluation, we've tried both fine-tuning and a Reward Model, but again, only the fine-tuned versions are available on Hugging Face. This is purely due to the fact that the fine-tuned models performed much better in both cases.
