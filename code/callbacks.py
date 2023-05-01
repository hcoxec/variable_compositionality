import sys
import pathlib
import os
import torch
import re
import wandb
import json

from datetime import datetime
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Union
from os import path, walk, makedirs, remove


from analyser import Analyser
from utils import registry, Interaction, Callback



class Checkpoint(NamedTuple):
    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    optimizer_scheduler_state_dict: Optional[Dict[str, Any]]


class CheckpointSaver(Callback):
    def __init__(
        self,
        checkpoint_path: Union[str, pathlib.Path],
        checkpoint_freq: int = 1,
        prefix: str = "",
        max_checkpoints: int = sys.maxsize,
    ):
        """Saves a checkpoint file for training.
        :param checkpoint_path:  path to checkpoint directory, will be created if not present
        :param checkpoint_freq:  Number of epochs for checkpoint saving
        :param prefix: Name of checkpoint file, will be {prefix}{current_epoch}.tar
        :param max_checkpoints: Max number of concurrent checkpoint files in the directory.
        """
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.checkpoint_freq = checkpoint_freq
        self.prefix = prefix
        self.max_checkpoints = max_checkpoints
        self.epoch_counter = 0

    def on_epoch_end(self, loss: float, logs, epoch: int):
        self.epoch_counter = epoch
        if self.checkpoint_freq > 0 and (epoch % self.checkpoint_freq == 0):
            filename = f"{self.prefix}_{epoch}" if self.prefix else str(epoch)
            self.save_checkpoint(filename=filename)

    def on_train_end(self):
        self.save_checkpoint(
            filename=f"{self.prefix}_final" if self.prefix else "final"
        )

    def save_checkpoint(self, filename: str):
        """
        Saves the game, agents, and optimizer states to the checkpointing path under `<number_of_epochs>.tar` name
        """
        self.checkpoint_path.mkdir(exist_ok=True, parents=True)
        if len(self.get_checkpoint_files()) > self.max_checkpoints:
            self.remove_oldest_checkpoint()
        path = self.checkpoint_path / f"{filename}.tar"
        torch.save(self.get_checkpoint(), path)

    def get_checkpoint(self):
        optimizer_schedule_state_dict = None
        if self.trainer.optimizer_scheduler:
            optimizer_schedule_state_dict = (
                self.trainer.optimizer_scheduler.state_dict()
            )

        game = self.trainer.game
        return Checkpoint(
            epoch=self.epoch_counter,
            model_state_dict=game.state_dict(),
            optimizer_state_dict=self.trainer.optimizer.state_dict(),
            optimizer_scheduler_state_dict=optimizer_schedule_state_dict,
        )

    def get_checkpoint_files(self):
        """
        Return a list of the files in the checkpoint dir
        """
        return [name for name in os.listdir(self.checkpoint_path) if ".tar" in name]

    @staticmethod
    def natural_sort(to_sort):
        """
        Sort a list of files naturally
        E.g. [file1,file4,file32,file2] -> [file1,file2,file4,file32]
        """
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
        return sorted(to_sort, key=alphanum_key)

    def remove_oldest_checkpoint(self):
        """
        Remove the oldest checkpoint from the dir
        """
        checkpoints = self.natural_sort(self.get_checkpoint_files())
        os.remove(os.path.join(self.checkpoint_path, checkpoints[0]))

@registry.register("callback", "wandb")
class WandbLogger(Callback):
    def __init__(
        self,
        config,
        **kwargs,
    ):

        self.config = config
        
        #to allow different logs for the same seed a random id is appended
        rand_s = datetime.now().time().microsecond 
        if config.mode == 'train':
            wandb.init(
                project=config.wandb_name, 
                group=config.run_id, 
                id=f'seed_{config.seed}_{rand_s}',  
                job_type=config.mode,
                reinit=True,
                **kwargs)
            wandb.config.update(config.__dict__)

    @staticmethod
    def log_to_wandb(metrics: Dict[str, Any], commit: bool = False, **kwargs):
        wandb.log(metrics, commit=commit, **kwargs)

    def on_train_begin(self, trainer_instance):  # noqa: F821
        self.trainer = trainer_instance
        #wandb.watch(self.trainer.game, log="all")

    def on_batch_end(
        self, logs, loss: float, batch_id: int, is_training: bool = True
    ):
        pass


    def on_epoch_end(self, loss: float, logs, epoch: int):
        if self.trainer.distributed_context.is_leader:
            current_acc = logs.aux['acc'].mean()
            partial_acc = logs.aux['acc_or'].mean()
            self.log_to_wandb(
                {
                    "train_loss": loss, 
                    "epoch": epoch, 
                    'acc': current_acc, 
                    'acc_or':partial_acc
                }, 
                commit=True
            )

    def on_validation_end(self, loss: float, logs, epoch: int):
        if self.trainer.distributed_context.is_leader:
            current_acc = logs.aux['acc'].mean()
            partial_acc = logs.aux['acc_or'].mean()
            self.log_to_wandb(
                {
                    "validation_loss": loss, 
                    "epoch": epoch, 
                    'val_acc': current_acc, 
                    'val_acc_or':partial_acc
                }, 
                commit=False
            )

@registry.register("callback", "stream_analysis")
class StreamAnalysis(Callback):
    def __init__(self, streamer, train_data, test_data, config, all_data=None) -> None:
        super().__init__(streamer=streamer)
        self.train_data = train_data
        self.test_data = test_data
        self.args = config
        self.all_data = all_data

    def on_train_begin(self, trainer_instance):  # noqa: F821
        self.trainer = trainer_instance
        analyser = Analyser(
                self.args,
                self.trainer.game,
                self.train_data
            )
        analyser.get_interactions(self.train_data.tensors[0].to(self.args.device))
        self.on_epoch_end(loss=42, logs=analyser.interaction, epoch=0)

        

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        results = {}
        
        self.analyser = Analyser(
                self.args,
                self.trainer.game,
                self.train_data
            )

        
        self.analyser.get_interactions(self.train_data.tensors[0].to(self.args.device))
        signals = self.analyser.signals
            
        train_results = self.analyser.full_prob_analysis(
                self.train_data, 
                signals.tolist(), 
                n_gram_size=1,
                split_name=self.train_data.split
            )
        train_results[f"{self.train_data.split}_acc"] = logs.aux['acc'].mean().item()
        train_results[f"{self.train_data.split}_acc_or"] = logs.aux['acc_or'].mean().item()
        
        if epoch % self.args.topsim_freq == 0:
            train_results[f"topsim"] = self.analyser.top_sim(
                    signals.tolist(), 
                    self.train_data.tensors[0]
                )

        if epoch % self.args.posdis_freq == 0:
            train_results[f"posdis"] = self.analyser.pos_dis(
                    self.train_data.tensors[0],
                    signals
                )
                
        for data in self.test_data:
            self.analyser.get_interactions(data.tensors[0].to(self.args.device))
            train_results[f"{data.split}_acc"] \
                = self.analyser.interaction.aux['acc'].mean().item()
            train_results[f"{data.split}_acc_or"] \
                = self.analyser.interaction.aux['acc_or'].mean().item()

            
        results.update(train_results)
        self.save_in_stream(results, epoch)
                
        if self.args.use_wandb:
            wandb.log(results, commit=False)

        
    