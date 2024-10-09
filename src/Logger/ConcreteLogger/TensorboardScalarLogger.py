from src.Logger.AbsLogger import AbsLogger
from torch.utils.tensorboard import SummaryWriter

class TensorboardScalarLogger(AbsLogger):
    """
        Logger that logs scalar values to tensorboard
    """

    def __init__(self, name:str, writer:SummaryWriter):
        """
            Params:t
                name - values indentifier
        """

        self.name = name
        self.writer = writer
        self.history = {}
        self.history['base'] = []

    
    # @property
    # def name(self):
    #     return self.name
    
    # @name.setter
    # def name(self, name):
    #     self.name = name

    # @property
    # def history(self):
    #     return self.history
    
    # @history.setter
    # def history(self, history:dict):
    #     self.history = history

    def Log(self, value:float) -> None:
        """
            Performs the Log function of all composite elements.

            Check the abstract base class for more info.
        """

        self.history['base'].append(value)

        self.writer.add_scalar(tag = self.name, scalar_value=value, global_step=len(self.history['base']) - 1)

    def LogLastEntriesMean(self, N:int, postfix:str = '_epoch') -> None:
        """
            Performs the Log function of all composite elements.

            Check the abstract base class for more info.
        """

        if postfix not in self.history.keys():
            self.history[postfix] = []

        self.history[postfix].append(sum(self.history['base'][-N:]) / N)

        self.writer.add_scalar(tag = self.name + postfix, scalar_value=self.history[postfix][-1], global_step=len(self.history[postfix]) - 1)

    def __str__(self) -> str:
        return f"Tensorboard Scalar Logger"