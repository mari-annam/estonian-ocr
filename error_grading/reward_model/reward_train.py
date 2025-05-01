import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

model_path = "tartuNLP/Llammas"
device = "cuda" if torch.cuda.is_available() else "cpu"

### model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device) juhul, kui otse tartuNLP/Lammast testida
model = AutoPeftModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id  

### Välja kommenteerida kui testida otse tartuNLP/Lammast
score_weights = torch.load("score.original_module.pth", map_location="cpu", weights_only=True)
model.score.original_module.load_state_dict(score_weights)

model.eval()

def get_rewards(batch_texts):
    """Tokenizes input texts and returns batch reward scores."""
    inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    rewards = (logits[:, 1] - logits[:, 0]).tolist()  # Reward = logit[1] - logit[0]
    return rewards

csv_file = "batch_test_test.csv"
batch_size = 8  

df = pd.read_csv(csv_file)

prompt_template = """### Instruction:
Paranda OCR vead selles eestikeelses tekstis.

### Input:
{ocr_text}

### Response:
"""

ocr_texts = [prompt_template.format(ocr_text=row["ocr_text"]) + row["ocr_text"] for _, row in df.iterrows()]
pred_texts = [prompt_template.format(ocr_text=row["ocr_text"]) + row["prediction"] for _, row in df.iterrows()]
gt_texts = [prompt_template.format(ocr_text=row["ocr_text"]) + row["ground_truth"] for _, row in df.iterrows()]

all_texts = ocr_texts + pred_texts + gt_texts

all_rewards = []
for i in range(0, len(all_texts), batch_size):
    batch_texts = all_texts[i:i + batch_size]
    batch_rewards = get_rewards(batch_texts)
    all_rewards.extend(batch_rewards)

num_rows = len(df)
df["reward_ocr"] = all_rewards[:num_rows]
df["reward_pred"] = all_rewards[num_rows:2*num_rows]
df["reward_gt"] = all_rewards[2*num_rows:]

output_csv = "batch_test_new_results.csv"
df.to_csv(output_csv, index=False)