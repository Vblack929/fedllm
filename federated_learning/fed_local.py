import torch
import copy
from trl import SFTTrainer
from transformers import TrainerCallback, Trainer
from peft import get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig, get_peft_model


def get_fed_local_trainer(script_args,
                          fed_args,
                          model,
                          tokenizer,
                          training_args,
                          local_dataset,
                          formatting_prompts_func,
                          data_collator,
                          global_dict,
                          local_auxiliary,
                          global_auxiliary):
    """ 
    For now we only consider FedAvg
    """

    if fed_args.fed_alg in ["fedavg"] or (fed_args.fed_alg).startswith('local'):
        if fed_args.peft_method == 'sft':
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                max_seq_length=script_args.seq_length,
                train_dataset=local_dataset,
                formatting_func=formatting_prompts_func,
                data_collator=data_collator,
            )
        elif fed_args.peft_method == 'lora':
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                # Example modules to apply LoRA
                target_modules=["query", "key", "value"],
                modules_to_tune=None,
            )
            model = get_peft_model(model, lora_config)

            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=local_dataset,
                data_collator=data_collator,
            )

    else:
        raise ValueError(
            f"Unsupported federated learning algorithm: {fed_args.fed_alg}")
    return trainer
