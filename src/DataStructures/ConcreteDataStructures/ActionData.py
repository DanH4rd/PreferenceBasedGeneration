import torch

from src.DataStructures.AbsData import AbsData


class ActionData(AbsData):
    """
    Class for storing actions for loss calc
    """

    def __init__(self, actions: torch.tensor):
        """
        Parametres:
            actions - [B, D] tensor, B - batch size, D - action dim
        """
        self.actions = actions

        if len(self.actions.shape) != 2:
            raise Exception(f"Invalid action tensor shape: {self.actions.shape}")

    def __str__(self) -> str:
        return "Action Loss Data"
