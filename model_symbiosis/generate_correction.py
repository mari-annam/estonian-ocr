import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from huggingface_hub import login

login("HUGGINGFACE TOKEN")

# load model from hugging face
def load_fixer():
    config = PeftConfig.from_pretrained("mariannam/llammas-OCR-FT5k")
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto", torch_dtype="auto")
    model = PeftModel.from_pretrained(base_model, "mariannam/llammas-OCR-FT5k")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.padding_side = "left"
    return model, tokenizer

# load model and tokenizer
fixer_model, fixer_tokenizer = load_fixer()

# generate corrected text
def fix_ocr_text(ocr_text: str) -> str:
    prompt = f"""### Instruction:
Paranda OCR vead selles eestikeelses tekstis.

### Input:
{ocr_text}

### Response:
"""
    input_length = len(fixer_tokenizer(ocr_text, truncation=True)['input_ids'])
    max_new_tokens = input_length + 10
    inputs = fixer_tokenizer(prompt, return_tensors="pt").to(fixer_model.device)
    outputs = fixer_model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.1, top_k=40)
    return fixer_tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

df = pd.read_csv("ocr_with_cer.csv")

# decide if to fix example based on predicted cer
def maybe_fix(row):
    cer = row["predicted_cer"]
    if 2 <= cer:
      return fix_ocr_text(row["ocr_text"])
    elif cer >= 45:
      # put to separate file
	return None

# save results
df["prediction"] = df.apply(maybe_fix, axis=1)
df.to_csv("ocr_with_fix.csv", index=False)
