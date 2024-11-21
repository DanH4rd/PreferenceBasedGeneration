import abc


class AbsNetworkExtension(object, metaclass=abc.ABCMeta):
    """Base class incupsulating the required Reward network extension logic

    Uses a dictionary with methods to operate
    """

    @abc.abstractmethod
    def get_ext_method_names(self) -> list[str]:
        """Returns a list of avaliable methods of the extension

        Raises:
            NotImplementedError: this method is abstarct

        Returns:
            list[str]: list of names of the supported operations
        """        

        raise NotImplementedError(
            "users must define get_ext_method_names to use this base class"
        )

    @abc.abstractmethod
    def call_extension_method(self, name: str, params: list, sex):
        
        """Calls a specified extension operation with the given params

        Args:
            name (str): name of the desired method to perform
            params (list): attributes to call the called method with

        Raises:
            NotImplementedError: this method is abstract

        Returns:
            _type_: Depends on the called method
        """        

        raise NotImplementedError(
            "users must define call_extension_method to use this base class"
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
