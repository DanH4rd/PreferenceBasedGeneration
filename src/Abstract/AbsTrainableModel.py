import abc


class AbsTrainableModel(object, metaclass=abc.ABCMeta):
    """Base class for models that support training-time concerns:
    eval/train mode switching, weight freezing, and device placement.
    """

    @abc.abstractmethod
    def set_to_evaluation_mode(self) -> None:
        """Sets the model to evaluation mode."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_to_train_mode(self) -> None:
        """Sets the model to train mode."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_train_mode(self) -> bool:
        """Returns True if the model is currently in train mode."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_device(self, device) -> None:
        """Moves the model to the given device."""
        raise NotImplementedError

    @abc.abstractmethod
    def freeze(self) -> None:
        """Freezes the model's weights."""
        raise NotImplementedError

    @abc.abstractmethod
    def unfreeze(self) -> None:
        """Unfreezes the model's weights."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_frozen(self) -> bool:
        """Returns True if the model's weights are currently frozen."""
        raise NotImplementedError
