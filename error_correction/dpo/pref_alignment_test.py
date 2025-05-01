import json
import torch
from tqdm import tqdm
from transformers import LlamaTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from datasets import Dataset
import pandas as pd

model_dir = 'model/pref-align-1104'

test = pd.read_csv('ocr_data_checked_test.csv')
test = Dataset.from_pandas(test)

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
)

tokenizer = LlamaTokenizer.from_pretrained(model_dir)
tokenizer.padding_side = "left"
print(f"Pad token ID: {tokenizer.pad_token_id}, EOS token ID: {tokenizer.eos_token_id}")

# batch_size = 2
preds = []

for ocr_text in tqdm(test['ocr_text']):
    prompt =
        f"""Paranda OCR vead selles eestikeelses tekstis.
        {ocr_text}"""
        
    input_length = len(tokenizer(ocr_text, truncation=True)['input_ids'])
    max_new_tokens = input_length  + 10

    inputs = tokenizer(prompt, max_length=4096, return_tensors='pt', truncation=True, padding=True).to('cuda')

    # Generate output
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.1,
            top_k=40,
        )

    output_tokens = outputs.shape[1]
    print(f"output tokens: {output_tokens}")

    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    preds.append(pred)

results_df = test.to_pandas()
results_df['Prediction'] = preds
results_df.to_csv('results/pref-align-results-1204.csv', index=False)

print("Predictions saved to 'results/lammas_dpo_corrections.csv'")
