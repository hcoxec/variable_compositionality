import torch
import os
import sys

from datetime import datetime
from typing import Any, Iterable, List, Optional, Callable
from torch.utils.data import DataLoader

from utils import Interaction, Batch, move_to
'''
The code below is based on related code from the Facebook Research EGG
Repo found here: 
https://github.com/facebookresearch/EGG/blob/main/egg/core/trainers.py

Reused inline with the MIT license
'''

class Trainer:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """

    def __init__(
        self,
        config,
        streamer,
        game: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_data: DataLoader,
        optimizer_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        validation_data: Optional[DataLoader] = None,
        device: torch.device = None,
        callbacks = None,
        grad_norm: float = None,
        aggregate_interaction_logs: bool = True,
        wandb_project: str = None,
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param optimizer_scheduler: An optimizer scheduler to adjust lr throughout training
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.config = config
        self.streamer = streamer
        self.game = game
        self.optimizer = optimizer
        self.optimizer_scheduler = optimizer_scheduler
        self.train_data = train_data
        self.validation_data = validation_data
        self.validation_freq = config.validation_freq
        self.device = config.device if device is None else device

        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks if callbacks else []
        self.grad_norm = grad_norm
        self.aggregate_interaction_logs = aggregate_interaction_logs
        self.wandb_project = wandb_project
        self.scaler = None

        self.update_freq = config.update_freq            


        
        self.game.to(self.device)

        self.optimizer.state = move_to(self.optimizer.state, self.device)


    def eval(self, data=None):
        mean_loss = 0.0
        interactions = []
        n_batches = 0
        validation_data = self.validation_data if data is None else data
        self.game.eval()
        with torch.no_grad():
            for batch in validation_data:
                if not isinstance(batch, Batch):
                    batch = Batch(*batch)
                batch = batch.to(self.device)
                optimized_loss, interaction = self.game(*batch)
                
                interaction = interaction.to("cpu")
                mean_loss += optimized_loss

                for callback in self.callbacks:
                    callback.on_batch_end(
                        interaction, optimized_loss, n_batches, is_training=False
                    )

                interactions.append(interaction)
                n_batches += 1

        mean_loss /= n_batches
        full_interaction = Interaction.from_iterable(interactions)

        return mean_loss.item(), full_interaction

    def train_epoch(self):
        mean_loss = 0
        n_batches = 0
        interactions = []

        self.game.train()

        self.optimizer.zero_grad()

        for batch_id, batch in enumerate(self.train_data):
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to(self.device)

       
            optimized_loss, interaction = self.game(*batch)                

            if self.update_freq > 1:
                optimized_loss = optimized_loss / self.update_freq

            if self.scaler:
                self.scaler.scale(optimized_loss).backward()
            else:
                optimized_loss.backward()

            if batch_id % self.update_freq == self.update_freq - 1:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                if self.grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.game.parameters(), self.grad_norm
                    )
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            n_batches += 1
            mean_loss += optimized_loss.detach()
      
            interaction = interaction.to("cpu")

            for callback in self.callbacks:
                callback.on_batch_end(interaction, optimized_loss, batch_id)

            interactions.append(interaction)

        if self.optimizer_scheduler:
            self.optimizer_scheduler.step()

        mean_loss /= n_batches
        full_interaction = Interaction.from_iterable(interactions)

        return mean_loss.item(), full_interaction

    def train(self, n_epochs):
        print("===================================================")
        print(f"Initializing Training at: {datetime.now()}")
        print(f"Intending to Train on Device: {self.config.device}")
        print("===================================================")

        self.streamer.start_run()

        t_loss, v_loss = 0.0, 0.0
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, n_epochs): 
            print(f"{datetime.now()} Step: {epoch} ", end='')
            sys.stdout.flush()

            self.streamer.step = epoch+1
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch + 1)
            train_loss, train_interaction = self.train_epoch()
            self.streamer.add({'loss':train_loss})

            print(f"Acc_or: {train_interaction.aux['acc_or'].mean()} Loss: {train_loss}")
            t_loss = train_loss
            for callback in self.callbacks:
                callback.on_epoch_end(
                    train_loss, 
                    train_interaction, 
                    epoch + 1)

            validation_loss = validation_interaction = None

            if (
                self.validation_data is not None
                and self.validation_freq > 0
                and (epoch + 1) % self.validation_freq == 0
            ):
                for callback in self.callbacks:
                    callback.on_validation_begin(epoch + 1)

                validation_loss, validation_interaction = self.eval()
                for callback in self.callbacks:
                    callback.on_validation_end(
                        validation_loss, validation_interaction, epoch + 1
                    )
                    self.streamer.add({'val_loss':validation_loss})
            self.streamer.record_state('train')
            if self.should_stop:
                for callback in self.callbacks:
                    callback.on_early_stopping(
                        train_loss,
                        train_interaction,
                        epoch + 1,
                        validation_loss,
                        validation_interaction,
                    )
                break

        for callback in self.callbacks:
            callback.on_train_end()

    def load(self, checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        if checkpoint.optimizer_scheduler_state_dict:
            self.optimizer_scheduler.load_state_dict(
                checkpoint.optimizer_scheduler_state_dict
            )
        self.start_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f"# loading trainer state from {path}")
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob("*.tar"):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)

        else:
            print('No Existing Checkpoint Found')
