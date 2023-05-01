from ast import walk
from tabnanny import check
import torch
import torch.nn as nn

from string import ascii_lowercase, punctuation, digits, ascii_uppercase

from itertools import combinations
from Levenshtein import distance
import numpy as np
import scipy.stats as stats
import json
import wandb
from datetime import datetime


from torchvision import datasets, transforms
from torch import nn
from torch.nn import functional as F

from os import path, walk, makedirs

from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Union
from attr import asdict, define, make_class, Factory


from analyser import Analyser
from train import build_model
from utils import registry, setup_streamer




class Checkpoint(NamedTuple):
    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    optimizer_scheduler_state_dict: Optional[Dict[str, Any]]

class Evaluator(object):
    
    def __init__(self, config, streamer, game, is_notebook=False):
        self.config = config
        self.streamer = streamer
        self.game = game
        
        Dataset = registry.lookup("dataset", config.dataset)

        all_data = {}
        for split in config.all_splits:
            all_data[split] = Dataset(config, split, scaling=1)

        self.eval_data = [all_data[split] for split in config.eval_splits]

        self.eval_steps = range(
            config.eval_steps['start'],
            config.eval_steps['stop'],
            config.eval_steps['step'],
        )
        
        self.checkpoint_dir = '{}/{}/{}/{}/seed_{}'.format(
            config.logdir,
            config.dataset,
            config.project_name,
            config.run_id,
            config.seed
        )

        self.checkpoint_prefix = 'sd_{}_hd_{}_sg_{}'.format(
            config.seed,
            config.hidden_size,
            config.signal_len,
        )

        if config.use_wandb and not is_notebook:
            rand_s = datetime.now().time().microsecond

            wandb.init(
                #settings=wandb.Settings(start_method="fork"), 
                project=config.wandb_name, #self.opts.proj_name, 
                group=config.run_id, 
                id=f'seed_{config.seed}_{rand_s}',  
                job_type='eval',
                reinit=True)

            wandb.config.update(config)

    def eval(self):
        for eval_step in self.eval_steps:
            filepath = f"{self.checkpoint_dir}/{self.checkpoint_prefix}_{eval_step}.tar"
            if self.check_path(filepath):
                analyser = self.get_checkpoint(filepath)
                results = self.eval_step(analyser, self.current_epoch, self.eval_data)
                
                self.save_in_stream(results, step=eval_step)

    def check_path(self, filepath):
        exist = path.exists(filepath)
        if not exist:
            print(f'not found: {filepath}')
        return exist

    def final_eval(self):
        filepath =  f'{self.save_prefix}_final.tar'
        self.load_from_checkpoint(filepath)
        analyser = Analyser(self.args, self.game, self.eval_data[0])
        return analyser

    def get_checkpoint(self, filepath):
        checkpoint = self.load_from_checkpoint(filepath)
        analyser = Analyser(self.config, self.game, self.eval_data[0])
        return analyser


    def save_in_stream(self, results, step, commit=True):
        if commit:
            self.streamer.record(results, dtype='eval')
            if self.config.use_wandb and not self.is_notebook:
                wandb.log(data=results, step=self.current_epoch, commit=True)
        else:
            self.streamer.add(results)
            if self.config.use_wandb and not self.is_notebook:
                wandb.log(data=results, step=self.current_epoch, commit=False)

    def eval_step(self, analyser: Analyser, step: int, data: list):
        results = {}
        print("===================================================")
        print(f'{datetime.now()}  Evaluating Epoch: {step}')
        print("===================================================")
        for split_id, data in enumerate(data):
            split_results = {}
            split=data.split
            print(f"\t split: {split}")
            analyser.get_interactions(data.input_tensors)
            

            
            split_results[f'{split}_acc'] = analyser.interaction.aux['acc'].mean().item()
            split_results[f'{split}_acc_or'] = analyser.interaction.aux['acc_or'].mean().item()
            
            if 'variation' in self.config.metrics:
                split_results.update( 
                    analyser.full_prob_analysis(
                        data, 
                        analyser.signals.tolist(), 
                        n_gram_size=1,
                        split_name=split
                    )
                )

            if 'topsim' in self.config.metrics:
                split_results[f'{split}_topsim'] = analyser.top_sim(
                        analyser.signals.tolist(), 
                        data.input_tensors
                    )
                
            if 'posdis' in self.config.metrics:
                split_results[f'{split}_posdis'] = analyser.pos_dis(
                        data.input_tensors,
                        analyser.signals
                )
            
            for measure in split_results:
                print(f"\t  - {measure}: {split_results[measure]}")

            results.update(split_results)

        return results
        

    def save_analyser(self, analyser, step, split):
        path = f'{self.save_prefix}_{step}_{split}.eval'
        torch.save(analyser, path)


    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f"# loading trainer state from {path}")
        checkpoint = torch.load(path, map_location=torch.device('cpu')) 
    
        self.load(checkpoint)
    
    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.sender = self.game.sender
        self.receiver = self.game.receiver
        self.current_epoch = checkpoint.epoch

        




def main(config):
    game = build_model(config)
    streamer = setup_streamer(config)
    print("===================================================")
    print(f'Initializing Evaluation at: {datetime.now()}')
    print("===================================================")
    evaler = Evaluator(
        config = config,
        streamer=streamer,
        game = game
    )

    evaler.eval()

    print("===================================================")
    print(f'Evaluation Finished at: {datetime.now()}')
