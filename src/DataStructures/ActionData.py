from typing import override

import torch

from src.Abstract.AbsData import AbsData


class ActionData(AbsData):
    """Class for storing actions for loss calc"""

    def __init__(self, actions: torch.Tensor):
        """

        Args:
            actions (torch.Tensor): [B, D] tensor containing actions,
                                    B - batch size, D - action dim

        Raises:
            Exception: if given tensor's dimention length isn't 2
        """

        self._check_tensor_format(actions)
        self._actions = actions.detach()

    @property
    def actions(self) -> torch.Tensor:
        """Returns a clone of the stored actions tensor

        Returns:
            torch.Tensor: [B, D] tensor containing actions
        """
        return self._actions.clone()

    @override
    def clone(self) -> "ActionData":
        """Returns an independent deep copy of this object

        Returns:
            ActionData: new object holding a cloned actions tensor
        """

        return ActionData(actions=self.actions)

    def append(self, actions: torch.Tensor):
        """Appends actions to the existing actions tensor.

        Args:
            actions (torch.Tensor): tensor with actions to append
        """

        self._check_tensor_format(actions)
        self._actions = torch.concat([self._actions, actions.detach()], dim=0)

    def _check_tensor_format(self, action_tensor: torch.Tensor):

        if len(action_tensor.shape) != 2:
            raise Exception(
                f"Invalid action tensor shape: {action_tensor.shape}. Expected [B, D] shape."
            )

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """
        return "Action Loss Data"
