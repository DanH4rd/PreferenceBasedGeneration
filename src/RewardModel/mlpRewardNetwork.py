import torch
import torch.nn as nn
from torch import device
from dataclasses import dataclass

from src.Abstract.AbsRewardModel import AbsNetworkExtension, AbsRewardModel
from src.Abstract.AbsTrainer import AbsTrainer
from src.DataStructures.ActionData import ActionData
from src.utils import freeze_model, unfreeze_model


class mlpRewardNetwork(nn.Module, AbsRewardModel):
    """Implementaion of a simple mlp neural network"""
    
    @dataclass
    class Configuration:
        """dataclass for grouping constructor parametres
        """
        input_dim: int
        hidden_dim: int
        p: float = 0.5

    @staticmethod
    def CreateFromConfiguration(conf: Configuration):
        return mlpRewardNetwork(input_dim= conf.input_dim, 
                                 hidden_dim=conf.hidden_dim,
                                 p=conf.p)
    
    def __init__(self, input_dim, hidden_dim, p=0.5):
        """

        Args:
            input_dim (_type_): dimention of the network input
            hidden_dim (_type_): dimention of the hidden layers
            p (float, optional): probability value for dropout layers,
                placed after each non output hidden layer. Defaults to 0.5.
        """

        super(mlpRewardNetwork, self).__init__()
        self.main = nn.Sequential(
            # seg1
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=p),
            # seg2
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=p),
            # seg3
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=p),
            # reward calc
            nn.Linear(hidden_dim, 1),
            # nn.BatchNorm1d(1),
            # nn.Sigmoid(),
        )

        self.isFrozen = False
        self.isTrainMode = True

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Alias for self.get_rewards

        Args:
            x (_type_): input tensor for the model of shape [B, D].
                B - number of actions, D - dimention of action vectors

        Returns:
            torch.tensor: list of rewards for each action
        """
        return self.get_rewards(x)

    def get_rewards(self, data: ActionData) -> torch.tensor:
        """Runs list of actions through the model
        to get predicted rewards

        Args:
            data (ActionData): list of actions serving as input for the model

        Returns:
            torch.tensor: list of rewards for each action
        """

        return self.main(data.actions)

    def get_stable_rewards(self, data: ActionData):
        """Runs list of actions through the model
        in evaluation mode, disabling dropout
        layers and setting batch norm layers in
        operating mode and etc. If the model is in
        train mode, method will temporeraly set model
        to eval and then return it to train.

        Args:
            data (ActionData): list of actions serving as input for the model

        Returns:
            torch.tensor: list of rewards for each action
        """

        returnToTrainMode = False

        if self.is_train_mode():
            self.set_to_evaluaion_mode()
            returnToTrainMode = True

        rewards = self.main(data.actions)

        if returnToTrainMode:
            self.set_to_train_mode()

        return rewards

    def set_to_evaluaion_mode(self):
        """Sets the model mode to evaluation mode"""
        self.eval()
        self.isTrainMode = False

    def set_to_train_mode(self):
        """Sets the model mode to train mode"""
        self.train()
        self.isTrainMode = True

    def is_train_mode(self) -> bool:
        """Returns true if the model is in train mode, false otherwise

        Returns:
            bool
        """
        return self.isTrainMode

    def set_device(self, device: str | device) -> None:
        """Set the model to the provided device

        Args:
            device (str | device): device identifier to set to
        """
        self.to(device)

    def get_trainer(self) -> AbsTrainer:
        """
        Returns the trainer object compatible with the given network

        Returns:
            trainer object set up for the calling network

        """
        raise NotImplementedError(f"Trainer is absent for {str(self)}")

    def get_extension(self) -> AbsNetworkExtension:
        """
        Returns object realising specific methods for the calling network

        Returns:
            extension object for the calling network object

        """
        raise NotImplementedError(f"Extension for {str(self)} not specified")

    def freeze(self) -> None:
        """Turns off weight updates for the model"""
        freeze_model(self)
        self.isFrozen = True

    def unfreeze(self) -> None:
        """Turns on weight updates for the model"""
        unfreeze_model(self)
        self.isFrozen = False

    def is_frozen(self) -> bool:
        """Returns true if the model's weights are frozen, false otherwise

        Returns:
            bool
        """
        return self.isFrozen

    def __str__(self) -> str:
        return "MLP reward network"
