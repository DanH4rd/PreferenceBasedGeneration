from typing import override

import lightning as L
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from src.Abstract.AbsLoss import AbsLoss
from src.Abstract.AbsTrainableModel import AbsTrainableModel
from src.Abstract.AbsTrainableRewardModel import AbsTrainableRewardModel
from src.DataStructures import (
    ActionData,
    ActionPairsData,
    ActionPairsPrefPairsContainer,
    PreferencePairsData,
    TrainableActionData,
)
from src.Loss import LogLossDecorator
from src.MetricsLogger import TensorboardScalarLogger


class ptlLightningWrapper(L.LightningModule):
    """Abstract class of a wrapper for base torch models"""

    loss_func_obj: AbsLoss

    @staticmethod
    def _wrap_loss_with_logging(
        loss_func_obj: AbsLoss,
        loss_log_tensorboard_writer: SummaryWriter | None,
        loss_log_name: str | None,
    ) -> AbsLoss:
        """Wraps loss_func_obj in a LogLossDecorator if a tensorboard writer is provided.

        Raises:
            Exception: if loss_log_tensorboard_writer is provided, loss_log_name must be a valid string
        """
        if loss_log_tensorboard_writer is None:
            return loss_func_obj

        if loss_log_name is None or len(loss_log_name) == 0:
            raise Exception(
                f"If loss_log_tensorboard_writer is provided, loss_log_name must be a valid string, got {loss_log_name}"
            )

        return LogLossDecorator(
            lossObject=loss_func_obj,
            logger=TensorboardScalarLogger(
                name=loss_log_name, writer=loss_log_tensorboard_writer
            ),
        )


class ptLightningModelWrapper(ptlLightningWrapper):
    """Wrapper class to transform a basic torch module to torch-lightning module"""

    def __init__(
        self,
        model: AbsTrainableRewardModel,
        loss_func_obj: AbsLoss,
        loss_log_tensorboard_writer: SummaryWriter | None = None,
        loss_log_name: str | None = None,
    ):
        """
        Args:
            model (AbsTrainableRewardModel): basic torch module representing the model
            loss_func_obj (AbsLoss): loss function object to use for loss calculation
                used during training
            loss_log_tensorboard_writer (SummaryWriter | None, optional): tensorboard writer object to log loss values. Defaults to None.
            loss_log_name (str | None, optional): name of the loss value to log in tensorboard. Defaults to None.
        Raises:
            Exception: if loss_log_tensorboard_writer is provided, loss_log_name must be a valid string
        """

        super().__init__()
        self.model = model
        self.loss_func_obj = self._wrap_loss_with_logging(
            loss_func_obj, loss_log_tensorboard_writer, loss_log_name
        )

    @override
    def forward(self, x: ActionData) -> torch.Tensor:
        """Run an input through a model and return
        model's output

        Args:
            x (ActionData): input data for object

        Returns:
            torch.Tensor: return value of the model
        """
        return self.model.get_rewards(x)

    @override
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Performs a training step for a given batch

        Args:
            batch (_type_): data batch to perform an optimisation
            step with
            batch_idx (_type_): id of the batch (?)

        Returns:
            torch.Tensor: loss valaue for the current batch with grad
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


class ptLightningLatentWrapper(ptlLightningWrapper):
    """Pytorch lightning wrapper that treats a TrainableActionData object's
    actions as a model parameter and optimises it by maximising the
    predicted reward got from a passed reward model.

    trainable_action registers its actions as an nn.Parameter, so the
    trainer optimises it in place directly (no copy-back step); the caller's
    reference to the same instance sees the updated values immediately.

    Does not modify the passed reward model during optimising actions.
    """

    def __init__(
        self,
        trainable_action: TrainableActionData,
        reward_model: AbsTrainableModel,
        loss_func_obj: AbsLoss,
        loss_log_tensorboard_writer: SummaryWriter | None = None,
        loss_log_name: str | None = None,
    ):
        """
        Args:
            trainable_action (TrainableActionData): action to optimise for reward maximisation
            reward_model (AbsTrainableModel): model whose weights are frozen during action optimisation
            loss_func_obj (AbsLoss): loss function object to use for loss calculation
                used during training
            loss_log_tensorboard_writer (SummaryWriter | None, optional): tensorboard writer object to log loss values. Defaults to None.
            loss_log_name (str | None, optional): name of the loss value to log in tensorboard. Defaults to None.
        Raises:
            Exception: if loss_log_tensorboard_writer is provided, loss_log_name must be a valid string
        """
        super().__init__()

        self.rewardModel = reward_model
        self.trainable_action = trainable_action

        self.loss_func_obj = self._wrap_loss_with_logging(
            loss_func_obj, loss_log_tensorboard_writer, loss_log_name
        )

    @override
    def forward(self, x):
        """Empty function"""

        return None

    @override
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Calculate loss for actions using the passed loss
        function object and returns loss value

        Args:
            batch (_type_): batch of dummy data.
            batch_idx (_type_): id of the batch (?)

        Returns:
            torch.Tensor: loss value for list of actions with grad
        """

        loss = self.loss_func_obj.calculate_loss(self.trainable_action)

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
        training step. No copy-back needed: trainable_action's Parameter
        is optimised in place.
        """
        self.rewardModel.unfreeze()

        pass

    @override
    def configure_optimizers(self):
        """Creates a torch optimiser used for training

        Returns:
            torch.optim: torch optimiser object
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
