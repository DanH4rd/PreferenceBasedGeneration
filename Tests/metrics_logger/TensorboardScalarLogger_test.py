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


import os

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor

from src.DataStructures.ImageData import ImageData
from src.MetricsLogger.TensorboardImageLogger import TensorboardImageLogger
from src.MetricsLogger.TensorboardScalarLogger import TensorboardScalarLogger


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

    def test_tensorboard_image(self):

        writer = SummaryWriter(log_dir="Tests/metrics_logger/runs")
        logger = TensorboardImageLogger(name="test/test_image", writer=writer)

        image_data_list = []
        for root, dirs, files in os.walk("Tests\\metrics_logger\\images"):
            for f in files:
                image = pil_to_tensor(
                    Image.open(os.path.join(root, f)).convert("RGB")
                ).unsqueeze(0)
                image_data_list.append(ImageData(images=image))

        for image in image_data_list:
            logger.log(image)

        writer.close()

        assert len(logger.history["base"]) == 5
