import abc

import torch

from src.DataStructures.AbsData import AbsData
from src.RewardModel.AbsNetworkExtension import AbsNetworkExtension
from src.Trainer.AbsTrainer import AbsTrainer


class AbsRewardModel(object, metaclass=abc.ABCMeta):
    """Base class incupsulating the required Reward model logic
    """

    @abc.abstractmethod
    def get_rewards(self, data: AbsData) -> torch.Tensor:
        """Returns rewards for given actions in the current model mode (f.e. eval or train)

        Args:
            data (AbsData): data to generate rewards for

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            torch.Tensor: list of rewards for given objects
        """        

        raise NotImplementedError("users must define GetRewards to use this base class")

    @abc.abstractmethod
    def get_stable_rewards(self, data: AbsData) -> torch.Tensor:
        """Returns rewards for given actions using evaluation mode of the network

        Args:
            data (AbsData): data for which to generate rewards

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            torch.Tensor: list of rewards for given objects
        """        

        raise NotImplementedError(
            "users must define get_stable_rewards to use this base class"
        )

    @abc.abstractmethod
    def set_to_evaluaion_mode(self) -> None:
        """Sets the model to evaluation mode

        Raises:
            NotImplementedError: this method is abstract
        """        
        raise NotImplementedError(
            "users must define set_to_evaluaion_mode to use this base class"
        )

    @abc.abstractmethod
    def set_to_train_mode(self) -> None:
        """Sets the model to train mode

        Raises:
            NotImplementedError: this method is abstract
        """        
        raise NotImplementedError(
            "users must define set_to_train_mode to use this base class"
        )

    @abc.abstractmethod
    def is_train_mode(self) -> bool:
        """Returns true if the model is in train mode

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            bool
        """        

        raise NotImplementedError(
            "users must define is_train_mode to use this base class"
        )

    @abc.abstractmethod
    def set_device(self, device) -> None:
        """Sets the used device object for the network

        Args:
            device (_type_): object/str defining the device

        Raises:
            NotImplementedError: this method is abstract
        """        

        raise NotImplementedError("users must define set_device to use this base class")

    @abc.abstractmethod
    def get_trainer(self) -> AbsTrainer:
        """Returns the trainer object compatible with the given network

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            AbsTrainer: trainer that can train the reward model
        """        

        raise NotImplementedError("users must define get_trainer to use this base class")

    @abc.abstractmethod
    def get_extension(self) -> AbsNetworkExtension:
        """Returns object containing not standard methods for the reward model

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            AbsNetworkExtension: object realising additional functionality 
        """        

        raise NotImplementedError("users must define get_extension to use this base class")

    @abc.abstractmethod
    def freeze(self) -> None:
        """Freezes the model's weights

        Raises:
            NotImplementedError: this model is abstract
        """        

        raise NotImplementedError("users must define freeze to use this base class")

    @abc.abstractmethod
    def unfreeze(self, flag: bool) -> None:
        """Unfreezes the model's weights

        Raises:
            NotImplementedError: this model is abstract
        """        
        raise NotImplementedError("users must define unfreeze to use this base class")

    @abc.abstractmethod
    def is_frozen(self) -> bool:
        """Returns true if the model is frozen

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            bool
        """        

        raise NotImplementedError(
            "users must define is_frozen to use this base class"
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
