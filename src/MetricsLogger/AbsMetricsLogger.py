import abc


class AbsMetricsLogger(object, metaclass=abc.ABCMeta):
    """
    Base class incupsulating the required logic for writing the training metrics
    to the chosen format
    """

    @abc.abstractmethod
    def log(self, value) -> None:
        """
        Logs the metrics calculated from the given params
        """
        raise NotImplementedError("users must define SetLogger to use this base class")

    @abc.abstractmethod
    def log_last_entries_mean(self, N: int, postfix: str) -> None:
        """
        Logs the mean of the last N recorded metrics in a separate metric

        Parametres:
            N - number of last logged values to mean and log
            postfix - name posfix for logger for the mean version values metric name
        """
        raise NotImplementedError(
            "users must define LogLastEntriesMean to use this base class"
        )

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns string describing the object
        """
        raise NotImplementedError("users must define __str__ to use this base class")
