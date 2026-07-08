import abc
from typing import Any, Self


class AbsData(object, metaclass=abc.ABCMeta):
    """Base class for data containers

    Subclasses make their own deep copy of the data passed to their
    constructors, so the object owns independent state that callers can't
    corrupt by later mutating the tensor they passed in.

    Exception: TrainableActionData neither copies nor detaches — it wraps
    a live nn.Parameter under active optimisation, and doing either would
    sever the autograd graph the training loop needs to backprop into.
    See its docstring for details.
    """

    @abc.abstractmethod
    def __str__(self) -> str:
        """Returns string describing the object

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            str
        """

        raise NotImplementedError("users must define __str__ to use this base class")

    @abc.abstractmethod
    def clone(self) -> Self:
        """Returns an independent deep copy of this object

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            Self: new object holding cloned copies of the underlying tensors
        """

        raise NotImplementedError("users must define clone to use this base class")

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Makes copy.deepcopy(obj) delegate to clone()

        Args:
            memo (dict[int, Any]): standard deepcopy memo dict, unused since
                clone() always creates fresh tensors

        Returns:
            Self: new object holding cloned copies of the underlying tensors
        """

        return self.clone()
