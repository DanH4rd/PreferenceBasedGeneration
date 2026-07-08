from typing import override

import torch

from src.Abstract.AbsData import AbsData

from .ActionData import ActionData


class ActionPairsData(AbsData):
    """Class for storing action pairs"""

    def __init__(self, action_pairs: torch.Tensor):
        """

        Args:
            action_pairs (torch.Tensor): [B,2,D] tensor, B - batch size, D - action dim

        Raises:
            Exception: if the given tensor is not of the expected shape
        """

        if len(action_pairs.shape) != 3 or action_pairs.shape[1] != 2:
            raise Exception(f"Invalid action tensor shape: {action_pairs.shape}")

        self._action_pairs = action_pairs.detach()

    @classmethod
    def from_split_actions(
        cls, action1: ActionData, action2: ActionData
    ) -> "ActionPairsData":
        """Creates an ActionPairsData object from two ActionData objects

        Args:
            action1 (ActionData): first action data
            action2 (ActionData): second action data

        Returns:
            ActionPairsData: combined action pairs data
        """
        action_pairs = torch.stack([action1.actions, action2.actions], dim=1)
        return cls(action_pairs=action_pairs)

    @property
    def action_pairs(self) -> torch.Tensor:
        """Returns a clone of the stored action pairs tensor

        Returns:
            torch.Tensor: [B,2,D] tensor, B - batch size, D - action dim
        """
        return self._action_pairs.clone()

    @override
    def clone(self) -> "ActionPairsData":
        """Returns an independent deep copy of this object

        Returns:
            ActionPairsData: new object holding a cloned action pairs tensor
        """

        return ActionPairsData(action_pairs=self.action_pairs)

    def get_split_actions(self) -> tuple[ActionData, ActionData]:
        """Splits action pairs into two tensors of shape [B,D]

        Returns:
            tuple[ActionData, ActionData]: two tensors of shape [B,D]
        """
        action_pairs = self.action_pairs
        return ActionData(actions=action_pairs[:, 0, :]), ActionData(
            actions=action_pairs[:, 1, :]
        )

    def __str__(self) -> str:
        """Returns a string representing an object

        Returns:
            str
        """
        return "Action Pairs Data"
