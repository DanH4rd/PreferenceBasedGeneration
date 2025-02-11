import torch

from src.Abstract.AbsData import AbsData


class ActionData(AbsData):
    """Class for storing actions for loss calc"""

    def __init__(self, actions: torch.tensor):
        """

        Args:
            actions (torch.tensor): [B, D] tensor containing actions,
                                    B - batch size, D - action dim

        Raises:
            Exception: if given tensor's dimention length isn't 2
        """

        self.actions = actions

        self._check_tensor_format(self.actions)
        
    def append(self, actions:torch.tensor):
        """_summary_

        Args:
            actions (torch.tensor): tensor with actions to append
        """

        self._check_tensor_format(actions)

        torch.concat([self.actions, actions], dim = 0)
        
        self._check_tensor_format(actions)

    def _check_tensor_format(self, action_tensor:torch.tensor):

        if len(self.actions.shape) != 2:
            raise Exception(f"Invalid action tensor shape: {action_tensor.shape}")

        
    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """
        return "Action Loss Data"
