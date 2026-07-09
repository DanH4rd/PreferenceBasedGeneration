import abc

import torch

from src.DataStructures import ActionPairsData, PreferencePairsData


class AbsTrainer(object, metaclass=abc.ABCMeta):
    """Base class incupsulating the required logic for training an ML model"""

    # @abc.abstractmethod
    # def SetLogger(self, logger: AbsLogger) -> None:
    #     """
    #         Assigns a logger object for the trainer
    #     """
    #     raise NotImplementedError('users must define SetLogger to use this base class')

    @abc.abstractmethod
    def run_training(
        self,
        action_data: ActionPairsData,
        preference_data: PreferencePairsData,
        epochs: int,
        sample_weights: torch.Tensor | None = None,
    ) -> None:
        """Run training for the given number of epochs

        Args:
            action_data (ActionPairsData): action pairs used as train input
            preference_data (PreferencePairsData): preference labels for action_data
            epochs (int): natural number of training epochs to perform
            sample_weights (torch.Tensor | None, optional): [B] tensor weighting each
                pair's contribution to the loss (e.g. RoundsMemory's discount factor).
                Defaults to every pair weighted equally.

        Raises:
            NotImplementedError: this method is abstract
        """

        raise NotImplementedError(
            "users must define run_training to use this base class"
        )

    @abc.abstractmethod
    def __str__(self) -> str:
        """Returns string describing the object

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            str
        """
        raise NotImplementedError("users must define __str__ to use this base class")
