from src.DataStructures.AbsData import AbsData
import torch

class ActionPairsData(AbsData):
    """
        Class for storing actions for loss calc
    """

    def __init__(self, actions_pairs:torch.tensor):
        """
            Parametres:
                actions - [B,2, D] tensor, B - batch size, D - action dim
        """
        self.actions_pairs = actions_pairs
 
        if (
            len(self.actions_pairs.shape) != 3 
            or self.actions_pairs.shape[1] != 2
            ):
            
            raise Exception(f'Invalid action tensor shape: {self.actions_pairs.shape}')
 
        if self.actions_pairs.shape[1] != 2:
            raise Exception(f'Provided tensor doesn\'t contain pairs: {self.actions_pairs.shape}')

    
    def __str__(self) -> str:
       return "Action Loss Data"