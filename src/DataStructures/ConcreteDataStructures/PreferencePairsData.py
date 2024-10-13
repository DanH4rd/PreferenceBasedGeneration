import torch

from src.DataStructures.AbsData import AbsData


class PreferencePairsData(AbsData):

    def __init__(self, preference_pairs: torch.tensor):
        """
        Parametres:
            y - [B,2] tensor, B - batch size

        TODO:
            add check for legal values
        """
        self.preference_pairs = preference_pairs

        if len(self.preference_pairs.shape) != 2 or self.preference_pairs.shape[1] != 2:
            raise Exception(f"Invalid action tensor shape: {self.y.shape}")

        present_pairs = torch.unique(self.preference_pairs, dim=0)

        allowed_values = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.0]]
        ).to(present_pairs.device)

        for pair in present_pairs:
            if not (allowed_values == pair).all(dim=1).any():
                raise Exception(f"Unallowed preference value: {str(pair.cpu())}")

    def __str__(self) -> str:
        return "Pair Preference Data"
