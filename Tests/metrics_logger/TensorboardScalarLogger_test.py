import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

###################################################
###################################################
###################################################
###################################################


from torch.utils.tensorboard import SummaryWriter

from src.MetricsLogger.ConcreteMetricsLogger.TensorboardScalarLogger import (
    TensorboardScalarLogger,
)


class TestMetricsLogger:

    def test_tensorboard_scalar(self):

        writer = SummaryWriter(log_dir="Tests/metrics_logger/runs")
        logger = TensorboardScalarLogger(name="test/test_value", writer=writer)

        logger.log(1)
        logger.log(2)
        logger.log(3)
        logger.log_last_entries_mean(3)

        logger.log(4)
        logger.log(5)
        logger.log_last_entries_mean(2)

        writer.close()

        assert logger.history["base"] == [1, 2, 3, 4, 5]
        assert logger.history["_epoch"] == [2, 4.5]
