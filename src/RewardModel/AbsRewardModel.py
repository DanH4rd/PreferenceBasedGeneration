import abc

import torch

from src.DataStructures.AbsData import AbsData
from src.RewardModel.AbsNetworkExtension import AbsNetworkExtension
from src.Trainer.AbsTrainer import AbsTrainer


class AbsRewardModel(object, metaclass=abc.ABCMeta):
    """
    Base class incupsulating the required Reward network logic
    """

    @abc.abstractmethod
    def GetRewards(self, data: AbsData) -> torch.Tensor:
        """
        Returns rewards for given actions in the current model mode (eval or train)

        Parametres:
            data - data about input objects

        Returns:
            [N] tensor containing N rewards
        """
        raise NotImplementedError("users must define GetRewards to use this base class")

    @abc.abstractmethod
    def GetStableRewards(self, data: AbsData) -> torch.Tensor:
        """
        Returns rewards for given actions using evaluation mode of the network

        Parametres:
            data - data about input objects

        Returns:
            [N] tensor containing N rewards
        """
        raise NotImplementedError(
            "users must define GetStableRewards to use this base class"
        )

    @abc.abstractmethod
    def SetToEvaluaionMode(self) -> None:
        """
        Sets the model to eval mode

        """
        raise NotImplementedError(
            "users must define SetToEvaluaionMode to use this base class"
        )

    @abc.abstractmethod
    def SetToTrainMode(self) -> None:
        """
        Sets the model to train mode

        """
        raise NotImplementedError(
            "users must define SetToTrainMode to use this base class"
        )

    @abc.abstractmethod
    def IsTrainMode(self) -> None:
        """
        Returns if the model is in train mode

        """
        raise NotImplementedError(
            "users must define IsTrainMode to use this base class"
        )

    @abc.abstractmethod
    def SetDevice(self, device) -> None:
        """
        Sets the used device object for the network

        Parametres:
            device - object/str defining the device

        """
        raise NotImplementedError("users must define SetDevice to use this base class")

    @abc.abstractmethod
    def GetTrainer(self) -> AbsTrainer:
        """
        Returns the trainer object compatible with the given network

        Returns:
            trainer object set up for the calling network

        """
        raise NotImplementedError("users must define GetTrainer to use this base class")

    @abc.abstractmethod
    def GetExtension(self) -> AbsNetworkExtension:
        """
        Returns object realising specific methods for the calling network

        Returns:
            extension object for the calling network object

        """
        raise NotImplementedError("users must define SetDevice to use this base class")

    @abc.abstractmethod
    def Freeze(self) -> None:
        """
        Freezes weights

        """
        raise NotImplementedError("users must define Freeze to use this base class")

    @abc.abstractmethod
    def Unfreeze(self, flag: bool) -> None:
        """
        Unfreezes the weights

        """
        raise NotImplementedError("users must define Unfreeze to use this base class")

    @abc.abstractmethod
    def IsFrozen(self) -> None:
        """
        Returns if the model weights are frozen

        """
        raise NotImplementedError(
            "users must define IsTrainMode to use this base class"
        )

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns string describing the object
        """
        raise NotImplementedError("users must define __str__ to use this base class")
