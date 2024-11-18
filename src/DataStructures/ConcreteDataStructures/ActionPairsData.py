import torch

from src.DataStructures.AbsData import AbsData


class ActionPairsData(AbsData):
    """Class for storing action pairs"""

    def __init__(self, action_pairs: torch.tensor):
        """

        Args:
            action_pairs (torch.tensor): [B,2,D] tensor, B - batch size, D - action dim

        Raises:
            Exception: if the given tensor is not of the expected shape
        """

        self.action_pairs = action_pairs

        if (
            len(self.action_pairs.shape) != 3
            or self.action_pairs.shape[1] != 2
            or self.action_pairs.shape[1] != 2
        ):

            raise Exception(f"Invalid action tensor shape: {self.action_pairs.shape}")

    def __str__(self) -> str:
        """Returns a string representing an object

        Returns:
            str
        """
        return "Action Loss Data"
