from src.Loss.AbsLoss import AbsLoss
from src.Logger.AbsLogger import AbsLogger
from src.DataStructures.AbsData import AbsData
import torch

class LogLossDecorator(AbsLoss):
    """
        A decorator class, which applied, will additionally log the loss
        value using a provided logger
    """

    def __init__(self, lossObject:AbsLoss, logger:AbsLogger):
        """
            Params:
                genModel - generator model object
                logger - if provided, will use it to log the loss
        """

        self.lossObject = lossObject
        self.logger = logger


    def CalculateLoss(self, data:AbsData) -> torch.tensor:
        """
            Calculates the loss and logs its value.

            Parametres:
                X - [B, D] tensor, B - batch size, D - action dim


            Check the abstract base class for more info.
        """ 

        loss = self.lossObject.CalculateLoss(data=data)

        self.logger.Log(loss)

        return loss

    def __str__(self) -> str:
        return f"Log Loss Decorator of {str(self.lossObject)}"