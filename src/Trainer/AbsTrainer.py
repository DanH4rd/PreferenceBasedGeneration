import abc


class AbsTrainer(object, metaclass=abc.ABCMeta):
    """
    Base class incupsulating the required training logic
    """

    # @abc.abstractmethod
    # def SetLogger(self, logger: AbsLogger) -> None:
    #     """
    #         Assigns a logger object for the trainer
    #     """
    #     raise NotImplementedError('users must define SetLogger to use this base class')

    @abc.abstractmethod
    def run_training(self, epochs: int) -> None:
        """
        Run training for the given number of epochs

        Parametres:
            epochs - natural number of training epochs to perform
        """
        raise NotImplementedError("users must define SetLogger to use this base class")

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns string describing the object
        """
        raise NotImplementedError("users must define __str__ to use this base class")
