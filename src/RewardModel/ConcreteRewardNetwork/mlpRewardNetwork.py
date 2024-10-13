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
        return self.GetRewards(x)

    def GetRewards(self, data: ActionData):
        return self.main(data.actions)

    def GetStableRewards(self, data: ActionData):

        returnToTrainMode = False

        if self.IsTrainMode():
            self.SetToEvaluaionMode()
            returnToTrainMode = True

        rewards = self.main(data.actions)

        if returnToTrainMode:
            self.SetToTrainMode()

        return rewards

    def SetToEvaluaionMode(self):
        self.eval()
        self.isTrainMode = False

    def SetToTrainMode(self):
        self.train()
        self.isTrainMode = True

    def IsTrainMode(self):
        return self.isTrainMode

    def SetDevice(self, device: str | device) -> None:
        self.to(device)

    def GetTrainer(self) -> AbsTrainer:
        """
        Returns the trainer object compatible with the given network

        Returns:
            trainer object set up for the calling network

        """
        raise NotImplementedError(f"Trainer is absent for {str(self)}")

    def GetExtension(self) -> AbsNetworkExtension:
        """
        Returns object realising specific methods for the calling network

        Returns:
            extension object for the calling network object

        """
        raise NotImplementedError(f"Extension for {str(self)} not specified")

    def Freeze(self) -> None:
        freeze_model(self)
        self.isFrozen = True

    def Unfreeze(self) -> None:
        unfreeze_model(self)
        self.isFrozen = False

    def IsFrozen(self) -> None:
        return self.isFrozen

    def __str__(self) -> str:
        return "MLP reward network"
