from src.Abstract.AbsMetricsLogger import AbsMetricsLogger


class CompositeLogger(AbsMetricsLogger):
    """Metrics Logger that is a composition of several other metrics loggers"""

    def __init__(self, loggers: list[AbsMetricsLogger]):
        """
        Args:
            loggers (list[AbsMetricsLogger]): list of metrics loggers that are
                elements of the composite logger
        """

        self.loggers = loggers

    def add_logger(self, logger: AbsMetricsLogger | list[AbsMetricsLogger]) -> None:
        """Adds a logger to the composite elements list. Can accept a list
        of loggers as a parametre, in this case it will concat
        the registered loggers list with the passed logger list

        Args:
            logger (AbsMetricsLogger | list[AbsMetricsLogger]): list of loggers or a logger to
                add to the composition elements
        """

        if isinstance(logger, list):
            self.loggers += logger
        else:
            self.loggers.append(logger)

    def log(self, value) -> None:
        """Performs the log function of all composite elements with the given value

        Args:
            value (_type_): value to log
        """

        for logger in self.loggers:
            logger.Log(value)

    def log_last_entries_mean(self, N: int, postfix: str = "_epoch") -> None:
        """Calls the log_last_entries_mean with given arguments on all
        composition elements

        Args:
            N (int): number of the newest elements in history to group and log
            postfix (str): identificator of the new aggregated value
        """

        for logger in self.loggers:
            logger.LogLastEntriesMean(N=N, postfix=postfix)

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """
        return f"Composite logger. Number of members: {len(self.loggers)}"
