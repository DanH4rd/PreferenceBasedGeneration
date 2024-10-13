import torch

from src.DataStructures.AbsData import AbsData


class ActionPairsData(AbsData):
    """
    Class for storing actions for loss calc
    """

    def __init__(self, action_pairs: torch.tensor):
        """
        Parametres:
            actions - [B,2, D] tensor, B - batch size, D - action dim
        """
        self.action_pairs = action_pairs

        if len(self.action_pairs.shape) != 3 or self.action_pairs.shape[1] != 2:

            raise Exception(f"Invalid action tensor shape: {self.action_pairs.shape}")

        if self.action_pairs.shape[1] != 2:
            raise Exception(
                f"Provided tensor doesn't contain pairs: {self.action_pairs.shape}"
            )

    def __str__(self) -> str:
        return "Action Loss Data"
