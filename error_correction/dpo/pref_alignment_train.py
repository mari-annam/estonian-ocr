from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer
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
    prompt = f"""Paranda OCR vead selles eestikeelses tekstis.
{example["ocr_text"]}<s>"""
    
    chosen = example["chosen_response"]
    rejected = example["rejected_response"]
    
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main(args):
    config = load_config(args.config)

    model_name = 'tartuNLP/Llammas'
    output_dir = os.path.join("model", "pref-align-1104")

    with open(args.data, "r") as f:
        data = json.load(f)

    processed_data = [
    prompt_template({
        "ocr_text": record["ocr_text"], 
        "chosen_response": record["chosen_response"], 
        "rejected_response": record["rejected_response"]
    })
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
        load_in_4bit=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    tokenizer.pad_token_id = 0 
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    config["learning_rate"] = float(config["learning_rate"])
    train_args = DPOConfig(
        output_dir=output_dir,
        **config,
        loss_type="ipo",
        logging_steps=10,  # Log every 10 steps
        log_level="info"
    )
    trainer = DPOTrainer(
        model,
        ref_model=None,
        beta=0.1,
        args=train_args,
        max_seq_length=4096,
        train_dataset=train,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    
    training_info = {
        "loss_type": train_args.loss_type,
        "learning_rate": train_args.learning_rate,
        "total_steps": train_args.num_train_epochs * len(train),
        "loss_history": trainer.state.log_history  # Now contains actual loss values
    }
    info_file = os.path.join(output_dir, "training_info.json")
    with open(info_file, "w") as f:
        json.dump(training_info, f, indent=4)
    
    trainer.save_model(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instruction-tuning lammas-base")
    parser.add_argument("--config", type=str, help="Path to config")
    parser.add_argument("--data", type=str, help="Path to training data (JSON file)")
    args = parser.parse_args()

    main(args)