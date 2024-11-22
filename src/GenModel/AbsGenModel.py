import abc

from torch.distributions import Distribution

from src.DataStructures.AbsData import AbsData
from src.DataStructures.ConcreteDataStructures.ImageData import ImageData
from src.MetricsLogger.AbsMetricsLogger import AbsMetricsLogger


class AbsGenModel(object, metaclass=abc.ABCMeta):
    """Base class incupsulating the required logic for a generative model"""

    @abc.abstractmethod
    def generate(self, data: AbsData) -> AbsData:
        """Generates values based on provided data

        Args:
            data (AbsData): Data to base generation on

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            AbsData: generated data
        """

        raise NotImplementedError("users must define generate to use this base class")

    @abc.abstractmethod
    def sample_random_actions(self, N: int) -> AbsData:
        """Returns N samples of noise used as input for this gen model

        Args:
            N (int): number if noise samples to return

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            AbsData: noise samples
        """

        raise NotImplementedError(
            "users must define sample_random_actions to use this base class"
        )

    @abc.abstractmethod
    def get_input_noise_distribution(self) -> Distribution:
        """Returns torch.Distribution object describing distribution
        of input noise samples

        Raises:
            NotImplementedError: this methodn is abstract

        Returns:
            Distribution: noise samples torch distribution

        TODO:
            replace torch.Distribution with ActionDistribution
        """
        raise NotImplementedError(
            "users must define get_input_noise_distribution to use this base class"
        )

    @abc.abstractmethod
    def get_media_logger(self) -> AbsMetricsLogger:
        """Returns a default logger for logging generated media

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            AbsLogger: logger capable of loggin generated data
        """
        raise NotImplementedError(
            "users must define get_media_logger to use this base class"
        )

    @abc.abstractmethod
    def set_device(self, device) -> None:
        """Sets the used device object for the model

        Args:
            device (_type_): object/str defining the device

        Raises:
            NotImplementedError: this method is abstract
        """

        raise NotImplementedError("users must define set_device to use this base class")

    @abc.abstractmethod
    def __str__(self) -> str:
        """Returns string describing the object

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            str
        """
        raise NotImplementedError("users must define __str__ to use this base class")
