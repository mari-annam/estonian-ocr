# source: https://github.com/Shef-AIRE/llms_post-ocr_correction/blob/main/llama-2.py
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import argparse
import os
import json
import torch
import yaml

def load_config(file):
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    return config['llama-2']

def prompt_template(example):
    output = []
    for ocr, pred, cer in zip(example["ocr_text"], example["prediction"], example["grade_for_prediction"]):
        string = f"""### Instruction:
Kui suur on tõenäosus, et parandatud tekst on OCR tekstist parem? Tagasta tõenäosus täisarvulise protsendina.

### Input:
OCR TEKST: {ocr}

PARANDATUD TEKST: {pred}

### Response:
{cer}</s>"""
        output.append(string)
    return output

def main(args):
    config = load_config(args.config)


    model_name = "tartuNLP/Llammas"
    output_dir = os.path.join("model", "lammas-prediction-grading")

    with open(args.data, "r") as f:
        data = json.load(f)

    processed_data = [
        {
            "ocr_text": record["ocr_text"],
            "prediction": record["prediction"],
            "grade_for_prediction": record["grade_for_prediction"],
        }
        for record in data
    ]

    train = Dataset.from_list(processed_data)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    tokenizer.pad_token_id = 0 
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    config["learning_rate"] = float(config["learning_rate"])

    train_args = SFTConfig(
        output_dir=output_dir,
        **config,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train,
        peft_config=peft_config,
        max_seq_length=4096,
        tokenizer=tokenizer,
        packing=False,
        formatting_func=prompt_template,
    )

    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training model to predict correction probability")
    parser.add_argument("--config", type=str, help="Path to config")
    parser.add_argument("--data", type=str, help="Path to training data (JSON file)")
    args = parser.parse_args()

    main(args)
