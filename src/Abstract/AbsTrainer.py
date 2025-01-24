import abc


class AbsTrainer(object, metaclass=abc.ABCMeta):
    """Base class incupsulating the required logic for training an ML model"""

    # @abc.abstractmethod
    # def SetLogger(self, logger: AbsLogger) -> None:
    #     """
    #         Assigns a logger object for the trainer
    #     """
    #     raise NotImplementedError('users must define SetLogger to use this base class')

    @abc.abstractmethod
    def run_training(self, epochs: int) -> None:
        """Run training for the given number of epochs

        Args:
            epochs (int): natural number of training epochs to perform

        Raises:
            NotImplementedError: this method is abstract
        """

        raise NotImplementedError(
            "users must define run_training to use this base class"
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
