from typing import List, Optional
import torch
import numpy as np
from loguru import logger
from models.plms import PLMModel
from defender import Defender
from data import wrap_dataset
from utils import evaluate_detection, evaluate_classification
from utils.evaluator import Evaluator
from .poisoners import load_poisoner
from trainers import load_trainer

class Attacker:
    """ 
    Each attacker has a poisoner and a trainer
    """
    def __init__(
        self, 
        poisoner: dict = {'name', 'base'},
        train: dict = {'name', 'base'},
        metrics: List[str] = ['accuracy'],
        **kwargs
    ):
        self.metrics = metrics
        self.poisoner = load_poisoner(poisoner)
        self.poison_trainer = load_trainer(dict(poisoner, **train, **{"poison_method":poisoner["name"]}))

        
    def attack(self, target: PLMModel, data: List, config: Optional[dict] = None, defender: Optional[Defender] = None):
        """
        Attack the target model with the attacker.

        Args:
            target (:obj:`target`): the target to attack.
            data (:obj:`List`): the dataset to attack.
            defender (:obj:`Defender`, optional): the defender.

        Returns:
            :obj:`target`: the attacked model.

        """
        poison_dataset = self.poison(target, data, "train")

        if defender is not None and defender.pre is True:
            # pre tune defense
            poison_dataset["train"] = defender.correct(poison_data=poison_dataset['train'])

        backdoored_model = self.train(target, poison_dataset)
        
        return backdoored_model
    
    def poison(self, target: PLMModel, dataset: List, mode: str):
        """
        Default poisoning function.

        Args:
            target (:obj:`target`): the target to attack.
            dataset (:obj:`List`): the dataset to attack.
            mode (:obj:`str`): the mode of poisoning. 
        
        Returns:
            :obj:`List`: the poisoned dataset.

        """
        return self.poisoner(dataset, mode)
    
    def train(self, target: PLMModel, dataset: List):
        """
        Use ``poison_trainer`` to attack the target model.
        default training: normal training

        Args:
            target (:obj:`target`): the target to attack.
            dataset (:obj:`List`): the dataset to attack.
    
        Returns:
            :obj:`target`: the attacked model.
        """
        return self.poison_trainer.train(target, dataset, self.metrics)
    
    def eval(self, target: PLMModel, dataset: List, defender: Optional[Defender] = None):
        """
        Default evaluation function (ASR and CACC) for the attacker.
            
        Args:
            target (:obj:`target`): the target to attack.
            dataset (:obj:`List`): the dataset to attack.
            defender (:obj:`Defender`, optional): the defender.

        Returns:
            :obj:`dict`: the evaluation results.
        """
        poison_dataset = self.poison(target, dataset, "eval")
        if defender is not None and defender.pre is False:
            
            if defender.correction:
                poison_dataset["test-clean"] = defender.correct(model=target, clean_data=dataset, poison_data=poison_dataset["test-clean"])
                poison_dataset["test-poison"] = defender.correct(model=target, clean_data=dataset, poison_data=poison_dataset["test-poison"])
            else:
                # post tune defense
                detect_poison_dataset = self.poison(target, dataset, "detect")
                detection_score, preds = defender.eval_detect(model=target, clean_data=dataset, poison_data=detect_poison_dataset)
                
                clean_length = len(poison_dataset["test-clean"])
                num_classes = len(set([data[1] for data in poison_dataset["test-clean"]]))
                preds_clean, preds_poison = preds[:clean_length], preds[clean_length:]
                poison_dataset["test-clean"] = [(data[0], num_classes, 0) if pred == 1 else (data[0], data[1], 0) for pred, data in zip(preds_clean, poison_dataset["test-clean"])]
                poison_dataset["test-poison"] = [(data[0], num_classes, 0) if pred == 1 else (data[0], data[1], 0) for pred, data in zip(preds_poison, poison_dataset["test-poison"])]


        poison_dataloader = wrap_dataset(poison_dataset, self.trainer_config["batch_size"])
        
        results = evaluate_classification(target, poison_dataloader, self.metrics)

        sample_metrics = self.eval_poison_sample(target, dataset, self.sample_metrics)

        return dict(results[0], **sample_metrics)
    
    def eval_poison_sample(self, target: PLMModel, dataset: List, eval_metrics=[]):
        """
        Evaluation function for the poison samples (PPL, Grammar Error, and USE).

        Args:
            target (:obj:`target`): the target to attack.
            dataset (:obj:`List`): the dataset to attack.
            eval_metrics (:obj:`List`): the metrics for samples. 
        
        Returns:
            :obj:`List`: the poisoned dataset.

        """
        evaluator = Evaluator()
        sample_metrics = {"ppl": np.nan, "grammar": np.nan, "use": np.nan}
        
        poison_dataset = self.poison(target, dataset, "eval")
        clean_test = self.poisoner.get_non_target(poison_dataset["test-clean"])
        poison_test = poison_dataset["test-poison"]

        for metric in eval_metrics:
            if metric not in ['ppl', 'grammar', 'use']:
                logger.info("  Invalid Eval Metric, return  ")
            measure = 0
            if metric == 'ppl':
                measure = evaluator.evaluate_ppl([item[0] for item in clean_test], [item[0] for item in poison_test])
            if metric == 'grammar':
                measure = evaluator.evaluate_grammar([item[0] for item in clean_test], [item[0] for item in poison_test])
            if metric == 'use':
                measure = evaluator.evaluate_use([item[0] for item in clean_test], [item[0] for item in poison_test])
            logger.info("  Eval Metric: {} =  {}".format(metric, measure))
            sample_metrics[metric] = measure
        
        return sample_metrics