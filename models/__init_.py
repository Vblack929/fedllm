import torch
import torch.nn as nn
from typing import List, Optional
from .base import Target
from .plms import PLMModel

Model_List = {
    'plm': PLMModel,
}


def load_target(config):
    target = Model_List[config["type"]](**config)
    return target

def mlm_to_seq_cls(mlm, config, save_path):
    mlm.plm.save_pretrained(save_path)
    config["type"] = "plm"
    model = load_target(config)
    model.plm.from_pretrained(save_path)
    return model