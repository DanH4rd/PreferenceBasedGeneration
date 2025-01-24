from torch.utils.tensorboard import SummaryWriter

from src.Abstract.AbsMetricsLogger import AbsMetricsLogger


class TensorboardScalarLogger(AbsMetricsLogger):
    """Logger that logs scalar values to tensorboard"""

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

    def log(self, value: float) -> None:
        """logs the passed value to tensorboard under
        the set name using the set writed

        Args:
            value (float): value to log
        """

        self.history["base"].append(value)

        self.writer.add_scalar(
            tag=self.name, scalar_value=value, global_step=len(self.history["base"]) - 1
        )

    def log_last_entries_mean(self, N: int, postfix: str = "_epoch") -> None:
        """Calculates the mean of the last N entries and logs a new value
        under a separate name (logger name + prefix)

        Args:
            N (int): number of the newest elements in history to group and log
            postfix (str): identificator of the new aggregated value.
                Defaults to "_epoch".

        Raises:
            AttributeError: if user tries to save an aggregated value to 'base' history
        """

        if postfix == "base":
            raise AttributeError('"base" postfix is reserved for standard logging')

        if postfix not in self.history.keys():
            self.history[postfix] = []

        self.history[postfix].append(sum(self.history["base"][-N:]) / N)

        self.writer.add_scalar(
            tag=self.name + postfix,
            scalar_value=self.history[postfix][-1],
            global_step=len(self.history[postfix]) - 1,
        )

    def get_logged_history(self, prefix="base") -> list:
        """Returns the list of all logged values.
        If given a prefix returns history of aggregated
        values associated with the given postfix

        Args:
            prefix (str, optional): _description_. Defaults to 'base'.

        Returns:
            list: history of logged values
        """

        return self.history[prefix]

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """

        return "Tensorboard Scalar Logger"
