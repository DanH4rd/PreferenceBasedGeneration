from src.Logger.AbsLogger import AbsLogger

class CompositeLogger(AbsLogger):
    """
        Logger that is a composition of several other loggers
    """

    def __init__(self, name:str, loggers:list[AbsLogger]):
        """
            Params:
                loggers - list of loggers out of which the composite consists of 
        """

        self.name = name
        self.loggers = loggers
        self.history = None


    def AddLogger(self, logger:AbsLogger|list[AbsLogger]) -> None:
        """
            Adds a logger to the composite elements list. Can accept a list
            of loggers as a parametre, in this case it will concat
            the registered loggers list with the passed logger lidt

            Params:
                loggers - AbsLogger object or a list of those
        """

        if isinstance(logger, list):
            self.loggers += logger
        else:
            self.loggers.append(logger)

    def Log(self, value) -> None:
        """
            Performs the Log function of all composite elements.

            Check the abstract base class for more info.
        """

        for logger in self.loggers:
            logger.Log(value)

    def LogLastEntriesMean(self, N:int, postfix:str = '_epoch') -> None:
        """
            Performs the Log function of all composite elements.

            Check the abstract base class for more info.
        """

        for logger in self.loggers:
            logger.LogLastEntriesMean(N=N, postfix=postfix)

    def __str__(self) -> str:
        return f"Composite logger. Number of members: {len(self.loggers)}"