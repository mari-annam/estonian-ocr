import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from huggingface_hub import login

login("HUGGINGFACE TOKEN")

# load model from hugging face
def load_improvement_estimator():
    config = PeftConfig.from_pretrained("mariannam/llammas-prediction-grading")
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto", torch_dtype="auto")
    model = PeftModel.from_pretrained(base_model, "mariannam/llammas-prediction-grading")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.padding_side = "left"
    return model, tokenizer

# load model and tokenizer
improvement_model, improvement_tokenizer = load_improvement_estimator()

# generate grade for corrected text
def estimate_improvement(ocr_text: str, fixed_text: str) -> int:
    prompt = f"""### Instruction:
Kui suur on tõenäosus, et parandatud tekst on OCR tekstist parem? Tagasta tõenäosus täisarvulise protsendina.

### Input:
OCR TEKST: {ocr_text}

PARANDATUD TEKST: {fixed_text}

### Response:
"""
    inputs = improvement_tokenizer(prompt, return_tensors="pt").to(improvement_model.device)
    outputs = improvement_model.generate(**inputs, max_new_tokens=3, temperature=0.7, top_p=0.1, top_k=40)
    prediction = improvement_tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    try:
        return int("".join(filter(str.isdigit, prediction)))
    except ValueError:
        return -1 # if the model does not return the correct output

df = pd.read_csv("ocr_with_fix.csv")

# find examples with bad grade
def improvement_filter(row):
    if pd.notnull(row["prediction"]):
        score = estimate_improvement(row["ocr_text"], row["prediction"])
        if score <= 40:
          # put bad corrections into a separate file to send for another round of corrections
        return prediction
    return None

# save results
df[["improvement_score"]] = df.apply(improvement_filter, axis=1)
df.to_csv("ocr_final_output.csv", index=False)
