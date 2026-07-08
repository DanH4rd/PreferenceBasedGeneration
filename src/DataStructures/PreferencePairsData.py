from typing import override

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

        preference_pairs = preference_pairs.detach()

        if len(preference_pairs.shape) != 2 or preference_pairs.shape[1] != 2:
            raise Exception(f"Invalid action tensor shape: {preference_pairs.shape}")

        # Probabilities of a pair must sum to 1, so we check if the sum of each pair is close to 1
        present_pairs = torch.unique(preference_pairs, dim=0)
        pair_vice_sum = present_pairs.sum(dim=1)

        too_big_total_probability = pair_vice_sum > 1.0 + torch.finfo(torch.float32).eps
        negative_values = (present_pairs < 0.0).any(dim=1)
        illegal_pairs = too_big_total_probability | negative_values

        if illegal_pairs.any():
            failed_pairs = present_pairs[illegal_pairs]
            raise Exception(
                f"Invalid preference pair values: {str(failed_pairs.cpu())}"
            )

        self._preference_pairs = preference_pairs

    @property
    def preference_pairs(self) -> torch.Tensor:
        """Returns a clone of the stored preference pairs tensor

        Returns:
            torch.Tensor: [B,2] tensor, B - batch size
        """
        return self._preference_pairs.clone()

    @override
    def clone(self) -> "PreferencePairsData":
        """Returns an independent deep copy of this object

        Returns:
            PreferencePairsData: new object holding a cloned preference pairs tensor
        """

        return PreferencePairsData(preference_pairs=self.preference_pairs)

    def __str__(self) -> str:
        """Returns a string representing an object

        Returns:
            str
        """
        return "Pair Preference Data"
