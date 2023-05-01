import torch

from typing import Dict, Iterable, Optional, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass(repr=True, unsafe_hash=True)
class Interaction:
    # incoming data
    sender_input: Optional[torch.Tensor]
    receiver_input: Optional[torch.Tensor]
    labels: Optional[torch.Tensor]
    aux_input: Optional[Dict[str, torch.Tensor]]

    # what agents produce
    message: Optional[torch.Tensor]
    receiver_output: Optional[torch.Tensor]

    # auxilary info
    message_length: Optional[torch.Tensor]
    aux: Dict[str, torch.Tensor]

    @property
    def size(self):
        interaction_fields = [
            self.sender_input,
            self.receiver_input,
            self.labels,
            self.message,
            self.receiver_output,
            self.message_length,
        ]
        for t in interaction_fields:
            if t is not None:
                return t.size(0)
        raise RuntimeError("Cannot determine interaction log size; it is empty.")

    def to(self, *args, **kwargs) -> "Interaction":
        """Moves all stored tensor to a device. For instance, it might be not
        useful to store the interaction logs in CUDA memory."""

        def _to(x):
            if x is None or not torch.is_tensor(x):
                return x
            return x.to(*args, **kwargs)

        self.sender_input = _to(self.sender_input)
        self.receiver_input = _to(self.receiver_input)
        self.labels = _to(self.labels)
        self.message = _to(self.message)
        self.receiver_output = _to(self.receiver_output)
        self.message_length = _to(self.message_length)

        if self.aux_input:
            self.aux_input = dict((k, _to(v)) for k, v in self.aux_input.items())
        if self.aux:
            self.aux = dict((k, _to(v)) for k, v in self.aux.items())

        return self

    @staticmethod
    def from_iterable(interactions: Iterable["Interaction"]) -> "Interaction":
        """
        >>> a = Interaction(torch.ones(1), None, None, {}, torch.ones(1), torch.ones(1), None, {})
        >>> a.size
        1
        >>> b = Interaction(torch.ones(1), None, None, {}, torch.ones(1), torch.ones(1), None, {})
        >>> c = Interaction.from_iterable((a, b))
        >>> c.size
        2
        >>> c
        Interaction(sender_input=tensor([1., 1.]), ..., receiver_output=tensor([1., 1.]), message_length=None, aux={})
        >>> d = Interaction(torch.ones(1), torch.ones(1), None, {}, torch.ones(1), torch.ones(1), None, {})
        >>> _ = Interaction.from_iterable((a, d)) # mishaped, should throw an exception
        Traceback (most recent call last):
        ...
        RuntimeError: Appending empty and non-empty interactions logs. Normally this shouldn't happen!
        """

        def _check_cat(lst):
            if all(x is None for x in lst):
                return None
            # if some but not all are None: not good
            if any(x is None for x in lst):
                raise RuntimeError(
                    "Appending empty and non-empty interactions logs. "
                    "Normally this shouldn't happen!"
                )
            return torch.cat(lst, dim=0)

        assert interactions, "interaction list must not be empty"
        has_aux_input = interactions[0].aux_input is not None
        for x in interactions:
            assert len(x.aux) == len(interactions[0].aux)
            if has_aux_input:
                assert len(x.aux_input) == len(
                    interactions[0].aux_input
                ), "found two interactions of different aux_info size"
            else:
                assert (
                    not x.aux_input
                ), "some aux_info are defined some are not, this should not happen"

        aux_input = None
        if has_aux_input:
            aux_input = {}
            for k in interactions[0].aux_input:
                aux_input[k] = _check_cat([x.aux_input[k] for x in interactions])
        aux = {}
        for k in interactions[0].aux:
            aux[k] = _check_cat([x.aux[k] for x in interactions])

        return Interaction(
            sender_input=_check_cat([x.sender_input for x in interactions]),
            receiver_input=_check_cat([x.receiver_input for x in interactions]),
            labels=_check_cat([x.labels for x in interactions]),
            aux_input=aux_input,
            message=_check_cat([x.message for x in interactions]),
            message_length=_check_cat([x.message_length for x in interactions]),
            receiver_output=_check_cat([x.receiver_output for x in interactions]),
            aux=aux,
        )

    @staticmethod
    def empty() -> "Interaction":
        return Interaction(None, None, None, {}, None, None, None, {})


class Batch:
    def __init__(
        self,
        sender_input: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        receiver_input: Optional[torch.Tensor] = None,
        aux_input = None,
    ):
        self.sender_input = sender_input
        self.labels = labels
        self.receiver_input = receiver_input
        self.aux_input = aux_input

    def __getitem__(self, idx):
        """
        >>> b = Batch(torch.Tensor([1]), torch.Tensor([2]), torch.Tensor([3]), {})
        >>> b[0]
        tensor([1.])
        >>> b[1]
        tensor([2.])
        >>> b[2]
        tensor([3.])
        >>> b[3]
        {}
        >>> b[6]
        Traceback (most recent call last):
            ...
        IndexError: Trying to access a wrong index in the batch
        """
        if idx == 0:
            return self.sender_input
        elif idx == 1:
            return self.labels
        elif idx == 2:
            return self.receiver_input
        elif idx == 3:
            return self.aux_input
        else:
            raise IndexError("Trying to access a wrong index in the batch")

    def __iter__(self):
        """
        >>> _ = torch.manual_seed(111)
        >>> sender_input = torch.rand(2, 2)
        >>> labels = torch.rand(2, 2)
        >>> batch = Batch(sender_input, labels)
        >>> it = batch.__iter__()
        >>> it_sender_input = next(it)
        >>> torch.allclose(sender_input, it_sender_input)
        True
        >>> it_labels = next(it)
        >>> torch.allclose(labels, it_labels)
        True
        """
        return iter(
            [self.sender_input, self.labels, self.receiver_input, self.aux_input]
        )

    def to(self, device: torch.device):
        """Method to move all (nested) tensors of the batch to a specific device.
        This operation doest not change the original batch element and returns a new Batch instance.
        """
        self.sender_input = move_to(self.sender_input, device)
        self.labels = move_to(self.labels, device)
        self.receiver_input = move_to(self.receiver_input, device)
        self.aux_input = move_to(self.aux_input, device)
        return self

def move_to(x: Any, device: torch.device) -> Any:
    """
    Simple utility function that moves a tensor or a dict/list/tuple of (dict/list/tuples of ...) tensors
        to a specified device, recursively.
    :param x: tensor, list, tuple, or dict with values that are lists, tuples or dicts with values of ...
    :param device: device to be moved to
    :return: Same as input, but with all tensors placed on device. Non-tensors are not affected.
             For dicts, the changes are done in-place!
    """
    if hasattr(x, "to"):
        return x.to(device)
    if isinstance(x, list) or isinstance(x, tuple):
        return [move_to(i, device) for i in x]
    if isinstance(x, dict) or isinstance(x, defaultdict):
        for k, v in x.items():
            x[k] = move_to(v, device)
        return x
    return x