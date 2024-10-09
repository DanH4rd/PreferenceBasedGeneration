import abc

class AbsLogger(object, metaclass=abc.ABCMeta):
    """
        Base class incupsulating the required logic for writing the training metrics
        to the chosen format
    """
    
    # @property
    # @abc.abstractmethod
    # def name(self):
    #     """
    #         String with loggers name. Used for indentifying the logged values
    #     """
    #     pass

    # @property
    # @abc.abstractmethod
    # def history(self):
    #     """
    #         Dictionary with logged values lists. Directly logged values are avaliable
    #         at the 'base' key. Values logged with LogLastEntriesMean are avaliable at
    #         the key equal to the used postfix (default 'on_epoch').
    #     """
    #     pass

    @abc.abstractmethod
    def Log(self, value) -> None:
        """
            Logs the metrics calculated from the given params
        """
        raise NotImplementedError('users must define SetLogger to use this base class')

    @abc.abstractmethod
    def LogLastEntriesMean(self, N:int, postfix:str) -> None:
        """
            Logs the mean of last N recorded metrics

            Parametres:
                N - number of last logged values to mean and log
                postfix - name posfix for logger for the mean version values
        """
        raise NotImplementedError('users must define LogLastEntriesMean to use this base class')

    @abc.abstractmethod
    def __str__(self) -> str:
        """
            Returns string describing the object
        """
        raise NotImplementedError('users must define __str__ to use this base class')