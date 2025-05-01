import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from datasets import Dataset
import pandas as pd
from huggingface_hub import login
import re


model_dir = 'model/lammas-prediction-grading'

test = pd.read_csv('prediction_grading_test.csv')
test = Dataset.from_pandas(test)

device = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoPeftModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.padding_side = "left"

batch_size = 8
preds = []

def is_numeric(text):
    """Check if the text contains only a numeric value, optionally followed by '%'."""
    return bool(re.fullmatch(r"\d+%?", text.strip()))

def generate_prediction(batch):
    """Generate model predictions."""
    prompts = [
        f"""### Instruction:
Kui suur on tõenäosus, et parandatud tekst on OCR tekstist parem? Tagasta tõenäosus täisarvulise protsendina.

### Input:
OCR TEKST: {ocr_text}

PARANDATUD TEKST: {predicted_text}

### Response:
""" for ocr_text, predicted_text in zip(batch['ocr_text'], batch['prediction'])
    ]

    inputs = tokenizer(prompts, max_length=4096, return_tensors='pt', truncation=True, padding=True).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=3,
            do_sample=True,
            temperature=0.7,
            top_p=0.1,
            top_k=40,
        )

    return [
        tokenizer.decode(output, skip_special_tokens=True)[len(prompt):].strip()
        for output, prompt in zip(outputs, prompts)
    ]

for i in tqdm(range(0, len(test), batch_size)):
    batch = test[i:i+batch_size]
    
    batch_preds = generate_prediction(batch)

    for j in range(len(batch_preds)):
        if not is_numeric(batch_preds[j]):  
            retry_pred = generate_prediction(Dataset.from_dict({
                'ocr_text': [batch['ocr_text'][j]],
                'predicted_text': [batch['prediction'][j]]
            }))[0]  
            batch_preds[j] = retry_pred if is_numeric(retry_pred) else "999"  # Assign "999" if still invalid

    preds.extend(batch_preds)

results_df = test.to_pandas()
results_df['grade_for_prediction'] = preds
results_df.to_csv('results/prediction-grading-results.csv', index=False)
