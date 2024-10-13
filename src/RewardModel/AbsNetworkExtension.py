import abc


class AbsNetworkExtension(object, metaclass=abc.ABCMeta):
    """
    Base class incupsulating the required Reward network  extension logic

    Uses a dictionary with methods to operate
    """

    @abc.abstractmethod
    def get_ext_method_names(self) -> list[str]:
        """
        Returns a list of avaliable methods

        Returns:
            list of strings
        """
        raise NotImplementedError(
            "users must define GetExtMethods to use this base class"
        )

    @abc.abstractmethod
    def call_extension_method(self, name: str, params: list) -> list[str]:
        """
        Calls an extension operation

        Returns:
            depends on the extension operation
        """
        raise NotImplementedError(
            "users must define CallExtensionMethod to use this base class"
        )

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns string describing the object
        """
        raise NotImplementedError("users must define __str__ to use this base class")
