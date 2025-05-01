import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from datasets import Dataset
import pandas as pd

# define the model directory
model_dir = 'model/lammas-ft13k' # change model name based on the model to use

# load the test data
test = pd.read_csv('ocr_data_checked_test.csv')
test = Dataset.from_pandas(test)

# configure the model to use 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# load the model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.padding_side = "left"
print(f"Pad token ID: {tokenizer.pad_token_id}, EOS token ID: {tokenizer.eos_token_id}")

# initialize the prediction list
preds = []

for ocr_text in tqdm(test['oldtext']):
    prompt = f"""### Instruction:
Paranda vead selles eestikeelses OCR tekstis.

### Input:
{ocr_text}

### Response:
"""

    # determine max_length dynamically based on full prompt length
    input_length = len(tokenizer(ocr_text, truncation=True)['input_ids'])
    max_new_tokens = input_length + 10
    # print(f"tokens allowed: {max_new_tokens}")

    # tokenize the input
    inputs = tokenizer(prompt, max_length=4096, return_tensors='pt', truncation=True, padding=True).to('cuda')

    # generate output
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.1,
            top_k=40,
            # repetition_penalty=1.15
        )

    output_tokens = outputs.shape[1]
    print(f"output tokens: {output_tokens}")

    # decode the output and strip prompts from the predictions
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    preds.append(pred)

# convert the results to a DataFrame
results_df = test.to_pandas()
results_df['prediction'] = preds

# save the results to a CSV file
results_df.to_csv('results/llammas-ft13k.csv', index=False) # change filename
