from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import AutoPeftModelForSequenceClassification
import pandas as pd

# Load model and tokenizer
model_path = "model/reward-eval-test"  # Update if needed
model = AutoPeftModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

score_weights = torch.load("score.original_module.pth", map_location="cpu", weights_only=True)
model.score.original_module.load_state_dict(score_weights)
print("Loaded trained score module state dict.")

model.eval()

def get_rewards(batch_texts):
    """Tokenizes input texts and returns batch reward scores."""
    inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    rewards = (logits[:, 1] - logits[:, 0]).tolist()  # Reward calculation
    return rewards

# Load data
csv_file = "reward_train.csv"  # Update with actual file
batch_size = 8  # Adjust batch size based on memory

df = pd.read_csv(csv_file)
df = df.head(10)  # Take first 10 rows for testing

# Prepare prompts
prompt_template = """### Instruction:
Paranda OCR vead selles eestikeelses tekstis.

### Input:
{ocr_text}

### Response:
"""

texts = [
    prompt_template.format(ocr_text=row["ocr_text"]) + row["ocr_text"]
    for _, row in df.iterrows()
] + [
    prompt_template.format(ocr_text=row["ocr_text"]) + row["prediction"]
    for _, row in df.iterrows()
] + [
    prompt_template.format(ocr_text=row["ocr_text"]) + row["ground_truth"]
    for _, row in df.iterrows()
]

# Process in batches
all_rewards = []
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]
    batch_rewards = get_rewards(batch_texts)
    all_rewards.extend(batch_rewards)

# Assign computed rewards back to dataframe correctly
num_rows = len(df)
df["reward_ocr"] = all_rewards[:num_rows]
df["reward_pred"] = all_rewards[num_rows:2*num_rows]
df["reward_gt"] = all_rewards[2*num_rows:]

# Save to CSV
df.to_csv("reward_output.csv", index=False)
print("Saved rewards to reward_output.csv")
