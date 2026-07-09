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
            Exception: if provided preferences are negative, or don't sum to 1
                (the [0., 0.] pair is a reserved exception meaning "skip")
        """

        preference_pairs = preference_pairs.detach()

        if len(preference_pairs.shape) != 2 or preference_pairs.shape[1] != 2:
            raise Exception(f"Invalid action tensor shape: {preference_pairs.shape}")

        present_pairs = torch.unique(preference_pairs, dim=0)

        pair_sum = present_pairs.sum(dim=1)
        sums_to_one = torch.isclose(pair_sum, torch.ones_like(pair_sum))
        is_skip = torch.isclose(present_pairs, torch.zeros_like(present_pairs)).all(
            dim=1
        )
        negative_values = (present_pairs < 0.0).any(dim=1)

        is_legal_pair = (sums_to_one | is_skip) & ~negative_values

        if not is_legal_pair.all():
            illegal_pairs = present_pairs[~is_legal_pair]
            raise Exception(
                f"Invalid preference pair values: {str(illegal_pairs.cpu())}"
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
