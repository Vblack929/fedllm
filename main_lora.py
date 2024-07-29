import copy
import os
from tqdm import tqdm
import numpy as np

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training
from datasets import load_dataset

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args


# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load IMDB dataset =====
dataset = load_dataset('imdb')
dataset = dataset['train'].train_test_split(test_size=0.2)

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = BertForSequenceClassification.from_pretrained(
    script_args.model_name_or_path,
    num_labels=2,
)
    
if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )


