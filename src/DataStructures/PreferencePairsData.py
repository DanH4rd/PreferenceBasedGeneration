import torch

from src.Abstract.AbsData import AbsData


class PreferencePairsData(AbsData):
    """Class for preference probabilities between a pair of objects"""

    def __init__(self, preference_pairs: torch.tensor):
        """
        Args:
            preference_pairs (torch.tensor): [B,2] tensor, B - batch size

        Raises:
            Exception: if provided tensor is of not expected shape
            Exception: if provided preferences contain values
                other than [1., 0.], [0., 1.].[0.5, 0.5] and [0., 0.]


        TODO:
            add check for legal values
            rename y argument to more descriptful name

            change valid preference data by checking if the pairs sum to 1
        """

        self.preference_pairs = preference_pairs

        if len(self.preference_pairs.shape) != 2 or self.preference_pairs.shape[1] != 2:
            raise Exception(f"Invalid action tensor shape: {self.y.shape}")

        # present_pairs = torch.unique(self.preference_pairs, dim=0)

        # allowed_values = torch.tensor(
        #     [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.0]]
        # ).to(present_pairs.device)

        # for pair in present_pairs:
        #     if not (allowed_values == pair).all(dim=1).any():
        #         raise Exception(f"Unallowed preference value: {str(pair.cpu())}")

    def __str__(self) -> str:
        """Returns a string representing an object

        Returns:
            str
        """
        return "Pair Preference Data"
