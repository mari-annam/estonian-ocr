import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from huggingface_hub import login

login("HUGGINGFACE TOKEN")

# load model from hugging face
def load_cer_estimator():
    config = PeftConfig.from_pretrained("mariannam/llammas-CER-prediction")
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto", torch_dtype="auto")
    model = PeftModel.from_pretrained(base_model, "mariannam/llammas-CER-prediction")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.padding_side = "left"
    return model, tokenizer

# load model and tokenizer
cer_model, cer_tokenizer = load_cer_estimator()

# generate cer
def get_cer(ocr_text: str) -> int:
    prompt = f"""### Instruction:
Kui suur protsent tähemärke sellest ajaloolisest eestikeelsest tekstist on vigane? Tagasta protsent täisarvuna.

### Input:
{ocr_text}

### Response:
"""
    inputs = cer_tokenizer(prompt, return_tensors="pt").to(cer_model.device)
    outputs = cer_model.generate(**inputs, max_new_tokens=3, temperature=0.7, top_p=0.1, top_k=40)
    prediction = cer_tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    try:
        return int("".join(filter(str.isdigit, prediction)))
    except ValueError:
        return -1 # if the model does not return the correct output

# save results
df = pd.read_csv("ocr_data_checked_test.csv")
df["predicted_cer"] = df["ocr_text"].apply(get_cer)
df.to_csv("ocr_with_cer.csv", index=False)
