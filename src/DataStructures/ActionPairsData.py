import torch

from src.Abstract.AbsData import AbsData
from src.DataStructures.ActionData import ActionData


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
        ):

            raise Exception(f"Invalid action tensor shape: {self.action_pairs.shape}")
        
    def get_split_actions(self) -> tuple[ActionData, ActionData]:
        """Splits action pairs into two tensors of shape [B,D]

        Returns:
            tuple[ActionData, ActionData]: two tensors of shape [B,D]
        """
        return ActionData(actions=self.action_pairs[:,0,:]), ActionData(actions=self.action_pairs[:,1,:])

    def __str__(self) -> str:
        """Returns a string representing an object

        Returns:
            str
        """
        return "Action Pairs Data"
