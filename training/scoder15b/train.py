"""
This script is used to fine-tune StarCoder2 family models on a java dataset, for code completion task.
"""
import torch 
from datasets import load_dataset, disable_caching
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, set_seed

# parallel processing
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=16)
from tqdm import tqdm
tqdm.pandas()

# utility
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

"""
Setting the variables.
"""

disable_caching()

set_seed(42) 

wproject = "name" # wb project name
run_name = "run-name" # name of the W&B run (optional)
# training batches
batch = 5
# Load base-model and tokenizer from HF-hub
checkpoint = "bigcode/starcoder2-15b"
# Select the column of interest from the dataset
text_column = 'content'

# training 
max_length = 1024
# model parallel
device_map = 'auto'

#wandb setup
import wandb
wandb.login()
os.environ["WANDB_PROJECT"] = wproject # wandb project name

"""
Loading the model and tokenizer
"""
# tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token # setting the pad token to the end of sequence token

# model
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, 
    device_map= device_map)


""""
Loading and preprocessing the data
"""
# LINK FOR THE DATASET: https://huggingface.co/datasets/AISE-TUDelft/memtune-tuning_data
# Load the data
dataset_train_20 =  load_dataset("AISE-TUDelft/memtune-tuning_data", name = "20k", split = 'train' )
dataset_valid_20 = load_dataset("AISE-TUDelft/memtune-tuning_data", name = "20k", split = 'valid' )

# Pick the columns of interest
train_20 = dataset_train_20['train'].select_columns(text_column)
validation_20 = dataset_valid_20['valid'].select_columns(text_column)

# Tokenize the sequences
# Note: StarCoder2 has a context lenght of 8,000 tokens,
def tokenize_input(batch):
    return tokenizer(batch[text_column], padding="max_length", truncation=True, max_length=max_length, return_tensors='pt')

training_20 = train_20.map(tokenize_input, batched=True, num_proc=64, remove_columns=text_column)
validating_20 = validation_20.map(tokenize_input, batched=True, num_proc=64,remove_columns=text_column)

""" 
Training initialization
"""
# Data collator
data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False, 
        return_tensors='pt'
    )


# Args
output_dir = "./epochs"
overwrite_output_dir= False

per_device_train_batch_size = batch
per_device_eval_batch_size = batch
gradient_accumulation_steps = 5

optim = "adafactor"
adam_beta1 = 0.9
weight_decay = 0.1 

learning_rate = 3e-5 
lr_scheduler_type = "linear" 
warmup_steps = 50

num_train_epochs = 3
eval_steps = 0.08 #200 # each epoch two evaluations
eval_strategy = "steps" # default is "no"
save_strategy = "epoch" # default is "steps"

logging_steps = 1
report_to = "wandb"

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir, 
    overwrite_output_dir=overwrite_output_dir,
    save_strategy = save_strategy,
    eval_strategy = eval_strategy,
    
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps = gradient_accumulation_steps,
    
    per_device_eval_batch_size=per_device_eval_batch_size,
    eval_steps = eval_steps,

    optim = optim,
    adam_beta1 = adam_beta1,
    weight_decay = weight_decay,
    
    learning_rate = learning_rate,
    lr_scheduler_type = lr_scheduler_type,
    warmup_steps = warmup_steps,
    
    logging_steps = logging_steps,
    report_to=report_to,
    run_name=run_name,
    seed = 42)

trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    train_dataset = training_20,
    eval_dataset = validating_20
)

# Training
trainer.train()