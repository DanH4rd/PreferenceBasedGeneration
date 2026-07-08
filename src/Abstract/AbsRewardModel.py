import abc

import torch

from src.DataStructures.ActionData import ActionData


class AbsRewardModel(object, metaclass=abc.ABCMeta):
    """Base class for reward models: maps actions to scalar reward predictions."""

    @abc.abstractmethod
    def get_rewards(self, data: ActionData) -> torch.Tensor:
        """Returns rewards for given actions in the current model mode.

        Args:
            data (ActionData): actions to generate rewards for

        Returns:
            torch.Tensor: reward for each action
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_stable_rewards(self, data: ActionData) -> torch.Tensor:
        """Returns rewards using evaluation mode (no dropout, stable batch norm).

        Args:
            data (ActionData): actions to generate rewards for

        Returns:
            torch.Tensor: reward for each action
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError
