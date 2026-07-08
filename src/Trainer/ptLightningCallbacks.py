from argparse import Namespace
from typing import Any, Dict, Optional, Union

import lightning as L
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.loggers.logger import Logger
from typing_extensions import override

from src.Loss import LogLossDecorator


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
        return 0

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

    def __init__(self, interval_length: int):
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
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module):
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
