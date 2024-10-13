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

from src.Logger.ConcreteLogger.TensorboardScalarLogger import TensorboardScalarLogger

writer = SummaryWriter(log_dir="UTs/logger_UT/runs")

logger = TensorboardScalarLogger(name="test/test_value", writer=writer)

logger.Log(1)
logger.Log(2)
logger.Log(3)
logger.LogLastEntriesMean(3)

logger.Log(4)
logger.Log(5)
logger.LogLastEntriesMean(2)

writer.close()

assert logger.history["base"] == [1, 2, 3, 4, 5]
assert logger.history["_epoch"] == [2, 4.5]
