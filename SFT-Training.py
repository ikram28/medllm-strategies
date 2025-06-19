import os
import sys
import logging
import argparse
import json

import torch
import datasets
import transformers
from transformers import set_seed, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer
import torch.distributed as dist
import idr_torch


logger = logging.getLogger(__name__)

os.environ['WANDB_MODE'] = 'offline'
wandb_project = "SFT-MistralNachos"


if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

logger = logging.getLogger(__name__)


def main():
    # CLI argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", type=str, help="HuggingFace model name")
    parser.add_argument("--path_train_dataset", type=str, default="./ft-data/data/train_data.json")
    parser.add_argument("--path_eval_dataset", type=str, default="./ft-data/data/test_data.json")
    parser.add_argument("--output_dir", type=str, default="./SFT-MistralNachos-models/")
    parser.add_argument("--logging_dir", type=str, default="./SFT-MistralNachos-logs/")
    parser.add_argument("--epochs", type=int, default=5, required=True)
    parser.add_argument("--batch_size", type=int, default=4, required=True)
    parser.add_argument("--save_steps", type=int, default=100, required=True)
    parser.add_argument("--logging_steps", type=int, default=10, required=True)
    parser.add_argument("--seed", type=int, default=42, required=True)
    parser.add_argument("--learning_rate", type=float, default=2e-5, required=True)

    args = parser.parse_args()

    # Training Arguments
    training_args = transformers.TrainingArguments(
        bf16=True,
        do_eval=True,
        eval_strategy="epoch",
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        push_to_hub=False,
        remove_unused_columns=True,
        report_to="wandb",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=100,
        seed=args.seed,
        #tf32=True,
        logging_dir=args.logging_dir,
        logging_first_step=True,
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
	    local_rank=idr_torch.local_rank,
    )

    set_seed(args.seed)

    # Logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")

    # Load data
    with open(args.path_train_dataset, 'r') as f:
        train_data = json.load(f)
    with open(args.path_eval_dataset, 'r') as f:
        val_data = json.load(f)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 2048

    # Define chat template 
    DEFAULT_CHAT_TEMPLATE = """
{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<|user|>\n' + message['content'] + eos_token }}
{% elif message['role'] == 'system' %}
{{ '<|system|>\n' + message['content'] + eos_token }}
{% elif message['role'] == 'assistant' %}
{{ '<|assistant|>\n' + message['content'] + eos_token }}
{% endif %}
{% if loop.last and add_generation_prompt %}
{{ '<|assistant|>' }}
{% endif %}
{% endfor %}
"""
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    # Apply chat template to dataset
    train_dataset = Dataset.from_dict({"chat": train_data})
    eval_dataset = Dataset.from_dict({"chat": val_data})
    train_dataset = train_dataset.map(lambda x: {
        "formatted_chat": tokenizer.apply_chat_template(
            x["chat"], tokenize=False, add_generation_prompt=False)
    })
    eval_dataset = eval_dataset.map(lambda x: {
        "formatted_chat": tokenizer.apply_chat_template(
            x["chat"], tokenize=False, add_generation_prompt=False)
    })

    ############
    # Model config
    ############
    logger.info("*** Load base model ***")

    model_kwargs = dict(
        trust_remote_code=True,
        use_flash_attention_2=True,
        torch_dtype=torch.float16,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map="auto"
    )

    # LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        modules_to_save=None,
        use_dora=True
    )

    # Trainer
    trainer = SFTTrainer(
        model=args.model_name,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="formatted_chat",
        tokenizer=tokenizer,
        packing=True,
        peft_config=peft_config,
        max_seq_length=tokenizer.model_max_length,
    )

    # Training
    train_result = trainer.train()

    # Save results
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_model(training_args.output_dir)

    # Optional: generate model card
    kwargs = {
        "finetuned_from": args.model_name,
        "dataset": args.output_dir,
        "dataset_tags": "Medical",
        "tags": ["Mistral", "Medical", "LLM"],
    }
    trainer.create_model_card(**kwargs)
    trainer.model.config.use_cache = True
    trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
