from typing import override

import lightning as L
import torch

from src.Abstract.AbsData import AbsData
from src.Abstract.AbsLoss import AbsLoss
from src.Abstract.AbsRewardModel import AbsRewardModel
from src.Abstract.AbsTrainableModel import AbsTrainableModel
from src.DataStructures.ActionData import ActionData
from src.DataStructures.ActionPairsData import ActionPairsData
from src.DataStructures.ActionPairsPrefPairsContainer import (
    ActionPairsPrefPairsContainer,
)
from src.DataStructures.PreferencePairsData import PreferencePairsData


class ptlLightningWrapper:
    """Abstract class of a wrapper for base torch models"""

    pass


class ptLightningModelWrapper(L.LightningModule, ptlLightningWrapper):
    """Wrapper class to transform a basic torch module to torch-lightning module"""

    def __init__(self, model: AbsRewardModel, loss_func_obj: AbsLoss):
        """
        Args:
            model (AbsRewardModel): basic torch module representing the model
            loss_func_obj (AbsLoss): loss function object to use for loss calculation
                used during training
        """

        super().__init__()

        self.model = model
        self.loss_func_obj = loss_func_obj

    @override
    def forward(self, x: AbsData) -> torch.tensor:
        """Run an input through a model and return
        model's output

        Args:
            x (AbsData): input data for object

        Returns:
            torch.tensor: return value of the model
        """
        return self.model(x)

    @override
    def training_step(self, batch, batch_idx) -> torch.tensor:
        """Performs a training step for a given batch

        Args:
            batch (_type_): data batch to perform an optimisation
            step with
            batch_idx (_type_): id of the batch (?)

        Returns:
            torch.tensor: loss valaue for the current batch with grad
        """

        t_pairs, t_prefs = batch

        t_pairs = t_pairs.to(self.device)
        t_prefs = t_prefs.to(self.device)

        x_b = t_pairs
        y_b = t_prefs

        data = ActionPairsPrefPairsContainer(
            action_pairs_data=ActionPairsData(action_pairs=x_b),
            pref_pairs_data=PreferencePairsData(preference_pairs=y_b),
        )

        loss = self.loss_func_obj.calculate_loss(data)

        return loss

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Creates a torch optimiser used for training

        Returns:
            torch.optim: torch optimiser object
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


class ptLightningLatentWrapper(L.LightningModule, ptlLightningWrapper):
    """Pytorch lightning wrapper that treats an Action Data object with a
    list of actions as a model and optimises it by maximising their
    predicted rewards got from a passed reward model.

    The updated action values after each train epoch are placed back
    into the originally passed ActionData object.

    Does not modify the passed reward model during optimising actions.
    """

    def __init__(
        self, action: ActionData, reward_model: AbsTrainableModel, loss_func_obj: AbsLoss
    ):
        """
        Args:
            action (ActionData): list of actions to optimise for reward maximisation
            reward_model (AbsTrainableModel): model whose weights are frozen during action optimisation
            loss_func_obj (AbsLoss): loss function object to use for loss calculation
                used during training
        """
        super().__init__()

        self.rewardModel = reward_model
        self.action_data_object = action
        self.action_data_object_device = action.actions.device
        self.action = torch.nn.parameter.Parameter(action.actions)

        self.loss_func_obj = loss_func_obj

    @override
    def forward(self, x):
        """Empty function"""

        return None

    @override
    def training_step(self, batch, batch_idx) -> torch.tensor:
        """Calculate loss for actions using the passed loss
        function object and returns loss value

        Args:
            batch (_type_): batch of dummy data.
            batch_idx (_type_): id of the batch (?)

        Returns:
            torch.tensor: loss value for list of actions with grad
        """

        t_pairs, t_prefs = batch

        data = ActionData(actions=self.action.to(self.device))

        loss = self.loss_func_obj.calculate_loss(data)

        return loss

    @override
    def on_train_epoch_start(self):
        """Freezes reward model's weights before the
        training step to not optimise the reward model's
        weights during training
        """
        self.rewardModel.freeze()
        pass

    @override
    def on_train_epoch_end(self):
        """Unfreezes back the reward model's weights after the
        training step as well as places the new values of actions
        back to the original ActionData object.
        """
        self.rewardModel.unfreeze()

        self.action_data_object.actions = self.action.data.detach().to(
            self.action_data_object_device
        )
        pass

    @override
    def configure_optimizers(self):
        """Creates a torch optimiser used for training

        Returns:
            torch.optim: torch optimiser object
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
