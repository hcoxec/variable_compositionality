import json
import wandb
import torch

from abc import ABC
from datetime import datetime
from os import path, walk, makedirs, remove
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Union

'''
This file defines a base class for callbacks and includes some general purpose
callbacks for checkpointing/saving results. Project specific callbacks are defined
in the directory for that project and instantiated via the registry
'''

class Callback(ABC):
    '''
    Note that while this is an abstract base class for all callbacks different
    training loops will pass in different arguments. For example Standard seq to seq training
    won't pass in signals. Bear this in mind when overwriting default functions
    '''

    def __init__(self, streamer=None) -> None:
        self.streamer = streamer

    def on_train_begin(self, trainer_instance):  # noqa: F821
        self.trainer = trainer_instance

    def on_train_end(self):
        pass

    def on_early_stopping(
        self,
        train_loss: float,
        train_logs,
        epoch: int,
        test_loss: float = None,
        test_logs = None,
    ):
        pass

    def on_validation_begin(self, epoch: int, **kwargs):
        pass

    def on_validation_end(self, loss: float, logs, epoch: int, **kwargs):
        pass

    def on_epoch_begin(self, epoch: int, **kwargs):
        pass

    def on_epoch_end(self, loss: float, logs, epoch: int, **kwargs):
        pass

    def on_batch_end(
        self, logs, loss: float, batch_id: int, 
        is_training: bool = True, **kwargs
    ):
        pass

    def save_in_stream(self, results, step=None):
        typed_results = {}
        for result in results:
            typed_results[result] = results[result].item() if (
                type(results[result]) == torch.Tensor
            ) else results[result]

        self.streamer.add(typed_results)
