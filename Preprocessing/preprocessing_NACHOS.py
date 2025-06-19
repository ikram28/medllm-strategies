import os
import json
import argparse
from itertools import chain

import pandas as pd
import pyarrow as pa
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer

# ---------------------
# Argument Parser Setup
# ---------------------
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model_name", type=str, required=True, default="mistralai/Mistral-7B-v0.1", help="HuggingFace model to use for tokenization")
parser.add_argument("--dataset", type=str, required=True, default="./dataset/huggingface_nachos_dataset/", help="Path to the dataset (in load_from_disk format)")
parser.add_argument("--subset", type=str, required=False, default=None, help="Subset to use if dataset is structured that way")
parser.add_argument("--output_dataset_path", type=str, required=True, default="./dataset/Out", help="Path to save the tokenized dataset")
parser.add_argument("--batch_size", type=int, required=False, default=10000, help="Batch size for grouping texts")
parser.add_argument("--seed", type=int, required=False, default=42, help="Random seed")
parser.add_argument("--preprocessing_num_workers", type=int, required=False, default=10, help="Parallelism for preprocessing")
args = parser.parse_args()

# ---------------------
# Load Dataset
# ---------------------
dataset = load_from_disk(args.dataset)
train_dataset = dataset["train"]  

# ---------------------
# Load Tokenizer
# ---------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=2048)
tokenizer.add_special_tokens({'pad_token': '<pad>'})

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def add_end_of_sentence(example):
    example["text"] = example["text"] + tokenizer.eos_token
    return example

print("Adding EOS token to each example...")
train_dataset = train_dataset.map(
    add_end_of_sentence,
    num_proc=args.preprocessing_num_workers,
    batched=False,
)

# ---------------------
# Tokenize Text
# ---------------------
def tokenize_function(example):
    return tokenizer(example["text"], return_special_tokens_mask=True)

print("Tokenizing dataset...")
tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    num_proc=args.preprocessing_num_workers,
    remove_columns=["text"],
    batched=True,
)

# ---------------------
# Group Texts into Blocks
# ---------------------
max_seq_length = 2048

def group_texts(examples):
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    if total_length >= max_seq_length:
        total_length = (total_length // max_seq_length) * max_seq_length
    result = {
        k: [t[i:i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated.items()
    }
    return result

print(f"Grouping texts into chunks of {max_seq_length} tokens...")
tokenized_datasets = tokenized_train_dataset.map(
    group_texts,
    batched=True,
    batch_size=args.batch_size,
    num_proc=args.preprocessing_num_workers,
    desc=f"Chunking sequences to length {max_seq_length}",
)

# ---------------------
# Filter for Exact-Length Sequences
# ---------------------
print("Filtering out sequences shorter than 2048 tokens...")
filtered_dataset = tokenized_datasets.filter(
    lambda example: len(example["input_ids"]) >= 2048,
    batched=False,
    num_proc=args.preprocessing_num_workers,
    desc="Filtering short sequences",
)

# ---------------------
# Save to Disk
# ---------------------
print(f"Saving dataset to: {args.output_dataset_path}")
filtered_dataset.save_to_disk(args.output_dataset_path)
print("✅ Dataset saved successfully.")
