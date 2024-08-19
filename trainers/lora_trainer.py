from .trainer import Trainer
from peft import LoraConfig, get_peft_model
from typing import Optional

class LoraTrainer(Trainer):
    def __init__(
		self,
		lora_r: int = 8,
		lora_alpha: int = 16,
		lora_dropout: float = 0.1,
		**kwargs,
	):
        super.__init__(**kwargs)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
    
    def register(self, model, dataloader, metrics):
        self.model = get_peft_model(
            model,
            LoraConfig(
                r=self.lora_r,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            ),
        )
        self.dataloader = dataloader
        self.metrics = metrics
        self.split_names = dataloader.keys()
        self.model.train()
        
        
        