from argparse import Namespace
from typing import Any, Dict, Optional, Union

import lightning as L
import torch
from lightning.pytorch.callbacks.callback import Callback
from pytorch_lightning.loggers.logger import Logger
from typing_extensions import override

from src.DataStructures.AbsData import AbsData
from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.DataStructures.ConcreteDataStructures.ActionPairsData import ActionPairsData
from src.DataStructures.ConcreteDataStructures.ActionPairsPrefPairsContainer import (
    ActionPairsPrefPairsContainer,
)
from src.DataStructures.ConcreteDataStructures.PreferencePairsData import (
    PreferencePairsData,
)
from src.Loss.AbsLoss import AbsLoss
from src.Loss.ConcreteLoss.LogLossDecorator import LogLossDecorator
from src.RewardModel.AbsRewardModel import AbsRewardModel
from src.Trainer.AbsTrainer import AbsTrainer


class ptlLightningWrapper:
    """Abstract class of a wrapper for base torch models
    """    
    pass


class ptLightningModelWrapper(L.LightningModule, ptlLightningWrapper):
    """Wrapper class to transform a basic torch module to torch-lightning module
    """    
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
    def forward(self, x:AbsData) -> torch.tensor:
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
        self, action: ActionData, reward_model: AbsRewardModel, loss_func_obj: AbsLoss
    ):
        """
        Args:
            action (ActionData): list of actions to optimise for reward maximisation
            reward_model (AbsRewardModel): model to get rewards for actions from
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
        """Empty function
        """        

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
        self.rewardModel.model.freeze()
        pass

    @override
    def on_train_epoch_end(self):
        """Unfreezes back the reward model's weights after the
        training step as well as places the new values of actions
        back to the original ActionData object. 
        """      
        self.rewardModel.model.unfreeze()

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


class ptLightningTrainer(AbsTrainer):
    """Implements the training logic for pytorch-lightning modules
    """    

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

        callbacks = [self.controller_callback]

        if not isinstance(model.loss_func_obj, LogLossDecorator):
            pass
        else:
            callbacks += [NotifyLossLoggerOnEpochEnd()]

        self.ptl_trainer = L.Trainer(
            enable_checkpointing=False,
            logger=TBLogger(),  # save_dir='.'),
            callbacks=callbacks,
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
            epochs (int): _description_

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
        """Returns string describing the object

        Returns:
            str
        """        
        return "PyTorch Lightning Trainer"


class TBLogger(Logger):
    """Empty pytorch lightning logger class to replace the default
    ptl logger in order to  disable automatic logging to tensorboard
    """
    
    def __init__(self):
        pass

    @property
    @override
    def name(self) -> Optional[str]:
        return "Empty Logger"

    @property
    @override
    def version(self) -> Optional[Union[int, str]]:
        return (0, "empty")

    @override
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Does nothing

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        pass

    @override
    def log_hyperparams(
        self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any
    ) -> None:
        """Does nothing.

        Args:
            params: :class:`~argparse.Namespace` or `Dict` containing the hyperparameters
            args: Optional positional arguments, depends on the specific logger being used
            kwargs: Optional keyword arguments, depends on the specific logger being used

        """
        pass


class EarlyStopAtEpochInterval(Callback):
    """A pytorch lightning callback that
    performes an early stop of the training process
    after the set number of training epochs. 

    Allows to run a single ptl trainer's fit
    method in parts.

    To resume training after pause trainer's 'should_stop'
    parametre should be set to False.
    """

    def __init__(self, interval_length:int):
        """
        Args:
            interval_length (int): number of epochs after which to
            invoke an early stop
        """        
        self.epoch_interval = interval_length
        self.epoch_counter = 0

    def set_epoch_interval(self, epoch_interval):
        """Sets the number of epochs after which to
            invoke an early stop

        Args:
            epoch_interval (_type_): number of epochs after which to
                invoke an early stop
        """        
        self.epoch_interval = epoch_interval

    def reset_epoch_counter(self):
        """Resets the counter traching the number of performed
        training epochs
        """        
        self.epoch_counter = 0

    @override
    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    @override
    def on_train_epoch_end(self, trainer:L.Trainer, pl_module):
        """On the end of the train epoch updates the counter and
        if the epoch_interval is reached early stoppes the training

        Args:
            trainer (_type_): the trainer object to which callback is attached
            pl_module (_type_): the pytorch lightning module that is being optimised (?)
        """        
        # trainer.current_epoch is currently finished
        # trainer.current_epoch + 1 is the next that will start
        # if (trainer.current_epoch + 1) % self.epoch_interval == 0:
        #     trainer.should_stop = True

        self.epoch_counter += 1

        if self.epoch_counter == self.epoch_interval:
            trainer.should_stop = True


class NotifyLossLoggerOnEpochEnd(Callback):
    """Callback that counts the number of batches in each train epoch 
    and calls LogLastEntriesMean fuction with this number for logger 
    corresponding to the loss object to log the aggregated metric value for an epoch.

    Loss object should be wrapped in LogLossDecorator to use this callback.
    """

    def __init__(self):
        self.counter = 0

    @override
    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    @override
    def on_train_batch_end(self, *args, **kwargs):
        self.counter += 1

    @override
    def on_train_epoch_end(self, trainer, pl_module):

        loss_func_obj = pl_module.loss_func_obj

        if not isinstance(loss_func_obj, LogLossDecorator):
            raise Exception(
                f"The Loss object ({str(loss_func_obj)}) is not wrapped in LogLossDecorator"
            )

        loss_func_obj.logger.log_last_entries_mean(self.counter)
        self.counter = 0
