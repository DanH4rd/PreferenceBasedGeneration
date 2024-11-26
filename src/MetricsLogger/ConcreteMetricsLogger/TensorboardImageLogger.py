from torch.utils.tensorboard import SummaryWriter

from src.MetricsLogger.AbsMetricsLogger import AbsMetricsLogger
from src.DataStructures.ConcreteDataStructures.ImageData import ImageData


class TensorboardImageLogger(AbsMetricsLogger):
    """Logger that logs image values to tensorboard"""

    def __init__(self, name: str, writer: SummaryWriter):
        """
        Args:
            name (str): identifier of the logged value
            writer (SummaryWriter): tensorboard writer to use for logging
        """

        self.name = name
        self.writer = writer
        self.history = {}
        self.history["base"] = []

    def log(self, value: ImageData) -> None:
        """logs the passed value to tensorboard under
        the set name using the set writed

        Args:
            value (float): value to log
        """

        if value.images.shape[0] > 1:
            raise Exception(f'Supported logging only for ine image at time, got {value.images.shape[0]}')
        
        self.history["base"].append(value)

        self.writer.add_image(tag=self.name, img_tensor=value.images[0], global_step=len(self.history["base"]) - 1)

    def log_last_entries_mean(self, N: int, postfix: str = "_epoch") -> None:
        """Empty function
        """
        pass

    def get_logged_history(self) -> list:
        """Returns the list of all logged values.

        Returns:
            list: history of logged values
        """

        return self.history['base']

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """

        return "Tensorboard Image Logger"
