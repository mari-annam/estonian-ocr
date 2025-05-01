import jiwer
import pandas as pd

file_path = "filename"
df = pd.read_csv(file_path)

def calculate_cer(prediction, target):
    """
    Calculate the Character Error Rate (CER) between a predicted text and the target text.
    """
    return jiwer.cer(target, prediction) * 100

def compute_wer(prediction, target):
    """
    Calculate the Word Error Rate (WER) between a target text and a prediction text.
    """
    return jiwer.wer(target, prediction) * 100

# calculate old cer between ocr text and ground truth and new cer between prediction and ground truth
df['cer_old'] = df.apply(lambda row: calculate_cer(str(row['ocr_text']), str(row['ground_truth'])), axis=1)
df['cer_new'] = df.apply(lambda row: calculate_cer(str(row['prediction']), str(row['ground_truth'])), axis=1)

# find cer change which is the difference between old cer and new cer
df['cer_change'] = df['cer_old'] - df['cer_new']

# find cer reduction
df['cer_reduction'] = df['cer_change'] / df['old_cer'] * 100

# calculate old wer between ocr text and ground truth and new wer between prediction and ground truth
df['wer_old'] = df.apply(lambda row: compute_wer(str(row['ocr_text']), str(row['ground_truth'])), axis=1)
df['wer_new'] = df.apply(lambda row: compute_wer(str(row['prediction']), str(row['ground_truth'])), axis=1)

# find wer change which is the difference between old wer and new wer
df['wer_change'] = df['wer_old'] - df['wer_new']

# find wer reduction
df['wer_reduction'] = df['wer_change'] / df['old_wer'] * 100

# filename to save the data into
output_path = f'{file_path}'
df.to_csv(output_path, index=False)
