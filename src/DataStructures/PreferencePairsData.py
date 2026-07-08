import torch

from src.Abstract.AbsData import AbsData


class PreferencePairsData(AbsData):
    """Class for preference probabilities between a pair of objects"""

    def __init__(self, preference_pairs: torch.Tensor):
        """
        Args:
            preference_pairs (torch.Tensor): [B,2] tensor, B - batch size

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
            raise Exception(
                f"Invalid action tensor shape: {self.preference_pairs.shape}"
            )

        # Probabilities of a pair must sum to 1, so we check if the sum of each pair is close to 1
        present_pairs = torch.unique(self.preference_pairs, dim=0)
        pair_vice_sum = present_pairs.sum(dim=1)

        too_big_total_probability = pair_vice_sum > 1.0 + torch.finfo(torch.float32).eps
        negative_values = (present_pairs < 0.0).any(dim=1)
        illegal_pairs = too_big_total_probability | negative_values

        if illegal_pairs.any():
            failed_pairs = present_pairs[illegal_pairs]
            raise Exception(
                f"Invalid preference pair values: {str(failed_pairs.cpu())}"
            )

    def __str__(self) -> str:
        """Returns a string representing an object

        Returns:
            str
        """
        return "Pair Preference Data"
