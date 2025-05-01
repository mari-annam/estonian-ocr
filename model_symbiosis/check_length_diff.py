import pandas as pd

df = pd.read_csv("round_3_pass.csv")

lower_threshold = 0.95
upper_threshold = 1.05

ocr_len = df["ocr_text"].str.len()
pred_len = df["prediction"].str.len()

mask = (pred_len >= lower_threshold * ocr_len) & (pred_len <= upper_threshold * ocr_len)
passed = df[mask]
failed = df[~mask]

failed.to_csv("need_new_predictions.csv", index=False)
passed.to_csv("passed.csv", index=False)
