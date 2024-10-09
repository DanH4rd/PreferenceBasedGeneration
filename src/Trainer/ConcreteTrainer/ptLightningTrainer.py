import lightning as L
from pytorch_lightning import loggers
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only
from lightning.pytorch.callbacks.callback import Callback


from typing_extensions import override
from typing import Any, Callable, Dict, Optional, Union
from argparse import Namespace

import torch

from src.Trainer.AbsTrainer import AbsTrainer
from src.RewardModel.AbsRewardModel import AbsRewardModel
from src.Loss.AbsLoss import AbsLoss
from src.Loss.ConcreteLoss.LogLossDecorator import LogLossDecorator
from src.DataStructures.ConcreteDataStructures.PairPreferenceData import PairPreferenceData
from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.DataStructures.ConcreteDataStructures.ActionPairsData import ActionPairsData
from src.DataStructures.ConcreteDataStructures.ActionPairsPrefPairsContainer import ActionPairsPrefPairsContainer
from src.DataStructures.ConcreteDataStructures.PairPreferenceData import PairPreferenceData

class ptlLightningWrapper():
    pass


class ptLightningModelWrapper(L.LightningModule, ptlLightningWrapper):
    def __init__(self, 
                 model:AbsRewardModel,
                 loss_func_obj:AbsLoss):
        super().__init__()

        self.model = model
        self.loss_func_obj = loss_func_obj

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        t_pairs, t_prefs = batch

        t_pairs = t_pairs.to(self.device)
        t_prefs = t_prefs.to(self.device)
        
        x_b = t_pairs
        y_b = t_prefs

        data = ActionPairsPrefPairsContainer(
            action_pairs_data=ActionPairsData(actions_pairs=x_b),
            pref_pairs_data=PairPreferenceData(y=y_b)
        )

        loss = self.loss_func_obj.CalculateLoss(data)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    

class ptLightningLatentWrapper(L.LightningModule, ptlLightningWrapper):
    def __init__(self, 
                 action: ActionData,
                 reward_model:AbsRewardModel,
                 loss_func_obj:AbsLoss):
        super().__init__()

        self.rewardModel = reward_model
        self.action = action.actions

        self.loss_func_obj = loss_func_obj

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        t_pairs, t_prefs = batch

        data = ActionData(
            actions=self.action
        )

        loss = self.loss_func_obj.CalculateLoss(data)
        
        return loss
    
    @override
    def on_train_epoch_start(self):
        self.rewardModel.model.Freeze()
        pass

    @override
    def on_train_epoch_end(self):
        self.rewardModel.model.Unfreeze()
        pass

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

class ptLightningTrainer(AbsTrainer):

    def __init__(self, 
                 model:ptlLightningWrapper, 
                 batch_size:int,
                 ):
        """
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
                                        logger=TBLogger(), #save_dir='.'), 
                                        callbacks=callbacks, 
                                        max_epochs=9999,
                                        enable_model_summary=False,
                                        enable_progress_bar=True
                                        )
        
        self.ptl_model = model


    def RunTraining(self, action_data:ActionPairsData, preference_data:PairPreferenceData, epochs:int) -> None:

        action_pair_tensor = action_data.actions_pairs
        preference_pair_tensor = preference_data.y

        if action_pair_tensor.shape[0] != preference_pair_tensor.shape[0]:
            raise Exception(f'Action pairs number and preference pairs number do not match: {action_pair_tensor.shape[0]} and {preference_pair_tensor.shape[0]}')

        train_ds = torch.utils.data.TensorDataset(action_pair_tensor, preference_pair_tensor)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size= self.batch_size, shuffle=True)

        self.controller_callback.setEpochInterval(epochs)
        self.controller_callback.resetEpochCounter()
        self.ptl_trainer.should_stop = False


        self.ptl_trainer.fit(self.ptl_model, train_dataloaders=train_dl)

        self.global_epoch += epochs

    def __str__(self) -> str:
        return 'PyTorch Lightning Trainer'
    



class TBLogger(Logger):
    """
        Empty pl lightning logger to disable automatic logging
        to tensorboard
    """

    def __init__(self):
        pass

    @property
    @override
    def name(self) -> Optional[str]:
        return 'Empty Logger'

    @property
    @override
    def version(self) -> Optional[Union[int, str]]:
        return (0, 'empty')

    @override
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Records metrics. This method logs metrics as soon as it received them.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded

        """
        pass

    @override
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any) -> None:
        """Record hyperparameters.

        Args:
            params: :class:`~argparse.Namespace` or `Dict` containing the hyperparameters
            args: Optional positional arguments, depends on the specific logger being used
            kwargs: Optional keyword arguments, depends on the specific logger being used

        """
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        return super().log_metrics(metrics, step)


class EarlyStopAtEpochInterval(Callback):
    """
        Allows to run pt lighting trainer training process
        by intervals and not all avaliable epochs at once.

        To resume training after pause trainer's 'should_stop'
        parametre should be set to False
    """

    def __init__(self, interval_length):
        self.epoch_interval = interval_length
        self.epoch_counter = 0

    def setEpochInterval(self, epoch_interval):
        self.epoch_interval = epoch_interval

    def resetEpochCounter(self):
        self.epoch_counter = 0

    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        # trainer.current_epoch is currently finished
        # trainer.current_epoch + 1 is the next that will start
        # if (trainer.current_epoch + 1) % self.epoch_interval == 0:
        #     trainer.should_stop = True
        self.epoch_counter += 1

        if self.epoch_counter == self.epoch_interval:
            trainer.should_stop = True



class NotifyLossLoggerOnEpochEnd(Callback):
    """
        Counts batch number for an epoch and calls LogLastEntriesMean
        fuction for logger corresponding to loss object.

        Loss object should be wrapped in LogLossDecorator.
    """

    def __init__(self):
        self.counter = 0

    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_batch_end(self, *args, **kwargs):
        self.counter += 1

    def on_train_epoch_end(self, trainer, pl_module):

        loss_func_obj = pl_module.loss_func_obj

        if not isinstance(loss_func_obj, LogLossDecorator):
            raise Exception(f'The Loss object ({str(loss_func_obj)}) is not wrapped in LogLossDecorator')

        loss_func_obj.logger.LogLastEntriesMean(self.counter)
        self.counter = 0
