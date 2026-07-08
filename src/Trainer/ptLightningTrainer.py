from dataclasses import dataclass

import lightning as L
import torch

from src.Abstract.AbsTrainer import AbsTrainer
from src.DataStructures.ActionPairsData import ActionPairsData
from src.DataStructures.PreferencePairsData import PreferencePairsData
from src.Loss.LogLossDecorator import LogLossDecorator
from src.Trainer.ptLightningCallbacks import (
    EarlyStopAtEpochInterval,
    NotifyLossLoggerOnEpochEnd,
    TBLogger,
)
from src.Trainer.ptLightningWrappers import ptlLightningWrapper


class ptLightningTrainer(AbsTrainer):
    """Implements the training logic for pytorch-lightning modules"""

    @dataclass
    class Configuration:
        """dataclass for grouping constructor parameters"""

        model: ptlLightningWrapper
        batch_size: int

    @staticmethod
    def create_from_configuration(conf: Configuration):
        return ptLightningTrainer(model=conf.model, batch_size=conf.batch_size)

    def __init__(
        self,
        model: ptlLightningWrapper,
        batch_size: int,
    ):
        """
        Args:
            model (ptlLightningWrapper): pytorch lightning module to train
            batch_size (int): size of batches into which to slice the train data
        TODO:
            add model optimiser parametrisation
        """

        self.global_epoch = 0

        self.controller_callback = EarlyStopAtEpochInterval(interval_length=5)
        self.batch_size = batch_size

        callbacks: list[L.Callback] = [self.controller_callback]

        if isinstance(model.loss_func_obj, LogLossDecorator):
            callbacks += [NotifyLossLoggerOnEpochEnd()]

        self.ptl_trainer = L.Trainer(
            enable_checkpointing=False,
            logger=TBLogger(),
            callbacks=callbacks,
            # TODO: make a workaround to make max epochs dynamically increase
            max_epochs=9999,
            enable_model_summary=False,
            enable_progress_bar=True,
        )

        self.ptl_model = model

    def run_training(
        self,
        action_data: ActionPairsData,
        preference_data: PreferencePairsData,
        epochs: int,
    ) -> None:
        """Runs the training process for a given number of epochs
        using action pairs list as input train data and preference data
        as true labels

        Args:
            action_data (ActionPairsData): list of action pairs used as train input
            preference_data (PreferencePairsData): preference list used as true labels
                for action pairs
            epochs (int): number of epochs to train for

        Raises:
            Exception: if number of action pairs does not match the preferences count
        """

        action_pair_tensor = action_data.action_pairs
        preference_pair_tensor = preference_data.preference_pairs

        if action_pair_tensor.shape[0] != preference_pair_tensor.shape[0]:
            raise Exception(
                f"Action pairs number and preference pairs number do not match: {action_pair_tensor.shape[0]} and {preference_pair_tensor.shape[0]}"
            )

        train_ds = torch.utils.data.TensorDataset(
            action_pair_tensor, preference_pair_tensor
        )
        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )

        self.controller_callback.set_epoch_interval(epochs)
        self.controller_callback.reset_epoch_counter()
        self.ptl_trainer.should_stop = False

        self.ptl_trainer.fit(self.ptl_model, train_dataloaders=train_dl)

        self.global_epoch += epochs

    def __str__(self) -> str:
        return "PyTorch Lightning Trainer"
