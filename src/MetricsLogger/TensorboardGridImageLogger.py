from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid

from src.Abstract.AbsMetricsLogger import AbsMetricsLogger
from src.DataStructures.ImageData import ImageData


class TensorboardGridImageLogger(AbsMetricsLogger):
    """Logger that assembles a batch of images into a grid and logs it to tensorboard.

    Accepts ImageData with any number of images, unlike TensorboardImageLogger
    which raises for batch size > 1.
    """

    def __init__(self, name: str, writer: SummaryWriter, nrow: int = 3):
        self.name = name
        self.writer = writer
        self.nrow = nrow
        self.step = 0

    def log(self, value: ImageData) -> None:
        grid = make_grid(value.images, nrow=self.nrow)
        self.writer.add_image(tag=self.name, img_tensor=grid, global_step=self.step)
        self.step += 1

    def log_last_entries_mean(self, N: int, postfix: str = "_epoch") -> None:
        pass

    def __str__(self) -> str:
        return f"Tensorboard Grid Image Logger ({self.name})"
