import torch.nn as nn
from torch import device

from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.RewardModel.AbsRewardModel import AbsNetworkExtension, AbsRewardModel
from src.Trainer.AbsTrainer import AbsTrainer
from src.utils import freeze_model, unfreeze_model


class mlpRewardNetwork(nn.Module, AbsRewardModel):

    def __init__(self, input_dim, hidden_dim, p=0.5):
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

    def forward(self, x):
        return self.get_rewards(x)

    def get_rewards(self, data: ActionData):
        return self.main(data.actions)

    def get_stable_rewards(self, data: ActionData):

        returnToTrainMode = False

        if self.is_train_mode():
            self.set_to_evaluaion_mode()
            returnToTrainMode = True

        rewards = self.main(data.actions)

        if returnToTrainMode:
            self.set_to_train_mode()

        return rewards

    def set_to_evaluaion_mode(self):
        self.eval()
        self.isTrainMode = False

    def set_to_train_mode(self):
        self.train()
        self.isTrainMode = True

    def is_train_mode(self):
        return self.isTrainMode

    def set_device(self, device: str | device) -> None:
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
        freeze_model(self)
        self.isFrozen = True

    def unfreeze(self) -> None:
        unfreeze_model(self)
        self.isFrozen = False

    def is_frozen(self) -> None:
        return self.isFrozen

    def __str__(self) -> str:
        return "MLP reward network"
