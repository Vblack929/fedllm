import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader

from datasets import Dataset, load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModel

class Client:
    def __init__(self, dataset,  optimizer, model, tokenizer, train_args):
        self.dataset = dataset
        self.optimizer = optimizer
        self.model = model
        self.tokenizer = tokenizer
        self.train_args = train_args
        