from torch.utils.tensorboard import SummaryWriter

from src.MetricsLogger.AbsMetricsLogger import AbsMetricsLogger


class TensorboardScalarLogger(AbsMetricsLogger):
    """
    Logger that logs scalar values to tensorboard
    """

    def __init__(self, name: str, writer: SummaryWriter):
        """
        Params:t
            name - values indentifier
        """

        self.name = name
        self.writer = writer
        self.history = {}
        self.history["base"] = []

    def log(self, value: float) -> None:
        """
        Performs the Log function of all composite elements.

        Check the abstract base class for more info.
        """

        self.history["base"].append(value)

        self.writer.add_scalar(
            tag=self.name, scalar_value=value, global_step=len(self.history["base"]) - 1
        )

    def log_last_entries_mean(self, N: int, postfix: str = "_epoch") -> None:
        """
        Calculates the mean of the last N entries and logs a new value
        under a separate name (logger name + prefix)

        New metrics are stored in the self.history under the prefix key

        Check the abstract base class for more info.
        """

        if postfix not in self.history.keys():
            self.history[postfix] = []

        self.history[postfix].append(sum(self.history["base"][-N:]) / N)

        self.writer.add_scalar(
            tag=self.name + postfix,
            scalar_value=self.history[postfix][-1],
            global_step=len(self.history[postfix]) - 1,
        )

    def __str__(self) -> str:
        return "Tensorboard Scalar Logger"
