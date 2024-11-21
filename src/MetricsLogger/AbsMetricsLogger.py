import abc


class AbsMetricsLogger(object, metaclass=abc.ABCMeta):
    """Base class incupsulating the required logic for writing the training metrics
    in the chosen way
    """

    @abc.abstractmethod
    def log(self, value) -> None:
        """Logs the the given value

        Args:
            value (_type_): value to log

        Raises:
            NotImplementedError: this method is abstract
        """
        raise NotImplementedError("users must define log to use this base class")

    @abc.abstractmethod
    def log_last_entries_mean(self, N: int, postfix: str) -> None:
        """takes the given number of last elements i history and logs
        an aggregated value of those elements.(f.e average of the last 5 logged
        accuracy values)

        Args:
            N (int): number of the newest elements in history to group and log
            postfix (str): identificator of the new aggregated value

        Raises:
            NotImplementedError: this method is abstract
        """

        raise NotImplementedError(
            "users must define log_last_entries_mean to use this base class"
        )

    @abc.abstractmethod
    def __str__(self) -> str:
        """Returns string describing the object

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            str
        """
        raise NotImplementedError("users must define __str__ to use this base class")
