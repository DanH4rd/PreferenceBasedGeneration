from src.DataStructures.AbsData import AbsData
import torch

class PairPreferenceData(AbsData):

    def __init__(self, y:torch.tensor):
        """
            Parametres:
                y - [B,2] tensor, B - batch size

            TODO:
                add check for legal values
        """
        self.y = y

        if (
                len(self.y.shape) != 2 
                or self.y.shape[1] != 2
           ):
            raise Exception(f'Invalid action tensor shape: {self.y.shape}')
        
        present_pairs = torch.unique(self.y, dim = 0)

        allowed_values = torch.tensor([[1.,0.], [0., 1.], [.5,.5], [0., 0.]]).to(present_pairs.device)

        for pair in present_pairs:
            if not (allowed_values == pair).all(dim=1).any() :
                raise Exception(f'Unallowed preference value: {str(pair.cpu())}')

    def __str__(self) -> str:
       return "Pair Preference Data"