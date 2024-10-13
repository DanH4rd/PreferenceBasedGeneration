import abc

from torch.distributions import Distribution

from src.DataStructures.AbsData import AbsData
from src.DataStructures.ConcreteDataStructures.ImageData import ImageData
from src.Logger.AbsLogger import AbsLogger


class AbsGenModel(object, metaclass=abc.ABCMeta):
    """
    Base class incupsulating the required logic for generative model
    """

    @abc.abstractmethod
    def generate(self, data: AbsData) -> ImageData:
        """
        Generates values based on provided data

        """
        raise NotImplementedError("users must define Generate to use this base class")

    @abc.abstractmethod
    def sample_random_actions(self, N: int) -> AbsData:
        """
        Returns N samples of noise used for generation
        """
        raise NotImplementedError("users must define GetNoise to use this base class")

    @abc.abstractmethod
    def get_input_noise_distribution(self) -> Distribution:
        """
        Returns the distribution over which generator operates
        """
        raise NotImplementedError("users must define GetNoise to use this base class")

    @abc.abstractmethod
    def get_media_logger(self) -> AbsLogger:
        """
        Returns a logger capable of logging the generated objects
        """
        raise NotImplementedError("users must define GetNoise to use this base class")

    @abc.abstractmethod
    def set_device(self, device) -> None:
        """
        Sets the used device object for the model

        Parametres:
            device - object/str defining the device
            isParralel - should
        """
        raise NotImplementedError("users must define SetDevice to use this base class")

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns string describing the object
        """
        raise NotImplementedError("users must define __str__ to use this base class")
