from typing import override

import torch
import torch.nn as nn

from .ActionData import ActionData


class TrainableActionData(ActionData, nn.Module):
    """ActionData variant that preserves the autograd graph of the given
    actions tensor instead of detaching it.

    Only meant for wrapping a grad-requiring actions tensor (e.g. an
    nn.Parameter being optimised) so a loss computed from it can backprop
    into that tensor. Everywhere else use plain ActionData.
    """

    error_msg = "TrainableActionData cannot be cloned because it is meant to be a single shared object with a single autograd graph. Use ActionData if you need a cloneable object."

    def __init__(self, actions: torch.Tensor):
        """

        Args:
            actions (torch.Tensor): [B, D] tensor containing actions,
                                    B - batch size, D - action dim

        Raises:
            Exception: if given tensor's dimention length isn't 2
        """

        nn.Module.__init__(self)
        self._check_tensor_format(actions)
        self._actions_param = torch.nn.parameter.Parameter(actions)

    @classmethod
    def from_action_data(cls, action_data: ActionData) -> "TrainableActionData":
        """Creates a TrainableActionData instance from an existing ActionData instance"""

        return cls(actions=action_data.actions)

    @override
    def clone(self) -> "TrainableActionData":

        raise NotImplementedError(self.error_msg)

    @property
    def actions(self) -> torch.Tensor:
        return self.grad_actions

    @property
    def grad_actions(self) -> torch.Tensor:
        """Returns the live, grad-tracked actions tensor.

        Used internally by the training loop's loss calculation to backprop
        into this object's Parameter. Do not use outside an active backward
        pass (e.g. for logging/inspection) — use detached_actions instead.

        Returns:
            torch.Tensor: [B, D] tensor containing actions, grad-tracked
        """
        return self._actions_param

    @property
    def detached_actions(self) -> torch.Tensor:
        """Returns a detached clone of the current actions tensor.

        Safe for logging, inspection, or any other non-training read.

        Returns:
            torch.Tensor: [B, D] tensor containing actions, detached
        """
        return self._actions_param.detach().clone()

    def append(self, actions: torch.Tensor):

        raise NotImplementedError(self.error_msg)

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """
        return "Trainable Action Data"
