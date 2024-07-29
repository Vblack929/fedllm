import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
from utils.merge_lora import merge_lora
from peft import get_peft_model, get_peft_model_state_dict, LoraConfig

# Load IMDB dataset and use a small subset for demo purposes
dataset = load_dataset("imdb", split="train[:1%]")
small_dataset = dataset.train_test_split(test_size=0.1)

base_model_name = "bert-base-uncased"
save_dir = "results/"

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

def get_lora_param_size(model):
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return lora_params

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenized_datasets = small_dataset.map(preprocess_function, batched=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize two BERT models
base_model = BertForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
model_1 = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# model_2 = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Apply LoRA to both models
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["classifier"],
)

model_1 = get_peft_model(model_1, lora_config)
# model_2 = get_peft_model(model_2, lora_config)

# Print trainable parameters for both models
print("Trainable parameters in model_1:")
model_1.print_trainable_parameters()
# print("\nTrainable parameters in model_2:")
# model_2.print_trainable_parameters()

# Define the trainer for model_1
trainer_1 = Trainer(
    model=model_1,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

# Train model_1
trainer_1.train()

# Display the size of added parameters by LoRA
lora_param_size_1 = get_lora_param_size(model_1)
# lora_param_size_2 = get_lora_param_size(model_2)

print(f"Size of LoRA parameters in model_1: {lora_param_size_1}")
# print(f"Size of LoRA parameters in model_2: {lora_param_size_2}")


# amerge lora
# merge_lora(base_model_name=base_model_name, lora_path=save_dir)

peft_model_params = get_peft_model_state_dict(model_1)
print(peft_model_params)

