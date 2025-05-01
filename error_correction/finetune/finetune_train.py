# source: https://github.com/Shef-AIRE/llms_post-ocr_correction/blob/main/llama-2.py
# source: https://github.com/TartuNLP/llammas
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import argparse
import os
import json
import torch
import yaml
from huggingface_hub import login

# load Llama 2 config from YAML file
def load_config(file):
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    return config['llama-2']


# format sample into prompt template
def prompt_template(example):
    output = []
    for ocr, gt in zip(example["ocr_text"], example["ground_truth"]):
        string = f"""### Instruction:
Paranda OCR vead selles eestikeelses tekstis.

### Input:
{ocr}

### Response:
{gt}</s>"""
        output.append(string)
    print(output)
    return output


# main function for instruction-tuning Llammas-base
def main(args):
    # load config
    config = load_config(args.config)

    login("HUGGINGFACE TOKEN")

    # select model for training (Llammas, Llammas-base, Llammas-GEC or Llammas finetune)
    model_name = "tartuNLP/Llammas"
    # model_name = "model/lammas-ocr-2"
    output_dir = os.path.join("model", "lammas-ft13k") # model name for saving

    # load training data from JSON
    with open(args.data, "r") as f:
        data = json.load(f)

    processed_data = [
        {"ocr_text": record["ocr_text"], "ground_truth": record["ground_truth"]}
        for record in data
    ]

    print(f"Number of training samples in processed data: {len(processed_data)}")

    train = Dataset.from_list(processed_data)
    for i, example in enumerate(train.select(range(10))):
        print(f"Example in dataset nr {i+1}: {example}")

    # quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # LoRA config
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # initialise model
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
    print(f"Pad token ID: {tokenizer.pad_token_id}, EOS token ID: {tokenizer.eos_token_id}")
    
    # instruction-tune model
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
    # Parse arguments for model/config/data
    parser = argparse.ArgumentParser(description="Instruction-tuning Llammas")
    parser.add_argument("--config", type=str, help="Path to config")
    parser.add_argument("--data", type=str, help="Path to training data (JSON file)")
    args = parser.parse_args()

    main(args)
