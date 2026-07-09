from typing import override

import torch

from src.Abstract.AbsData import AbsData

from .ActionPairsData import ActionPairsData
from .PreferencePairsData import PreferencePairsData


class ActionPairsPrefPairsContainer(AbsData):
    """Class for storing ActionPairsData and PreferencepairData together,
    with a per-pair sample weight (defaults to 1.0 for every pair) that
    scales each pair's contribution to the loss without touching the
    preference values themselves.
    """

    def __init__(
        self,
        action_pairs_data: ActionPairsData,
        pref_pairs_data: PreferencePairsData,
        sample_weights: torch.Tensor | None = None,
    ):
        """

        Args:
            action_pairs_data (ActionPairsData): list of action pairs
            pref_pairs_data (PreferencePairsData): list of preference pairs
            sample_weights (torch.Tensor | None, optional): [B] tensor weighting each
                pair's contribution to the loss (e.g. RoundsMemory's discount factor).
                Defaults to a tensor of ones (every pair weighted equally).

        Raises:
            Exception: if action_pairs_data doesn't have the same number
                of elements (value of the zero dim size) as pref_pairs_data
            Exception: if sample_weights doesn't have the same number of
                elements as action_pairs_data
        """

        if (
            action_pairs_data.action_pairs.shape[0]
            != pref_pairs_data.preference_pairs.shape[0]
        ):
            raise Exception(
                f"Action pairs number does not correspond to preference pairs number: {action_pairs_data.action_pairs.shape[0]} and {pref_pairs_data.preference_pairs.shape[0]}"
            )

        if sample_weights is None:
            sample_weights = torch.ones(action_pairs_data.action_pairs.shape[0])
        elif sample_weights.shape[0] != action_pairs_data.action_pairs.shape[0]:
            raise Exception(
                f"Sample weights count does not correspond to action pairs number: {sample_weights.shape[0]} and {action_pairs_data.action_pairs.shape[0]}"
            )

        self._action_pairs_data = action_pairs_data.clone()
        self._pref_pairs_data = pref_pairs_data.clone()
        self._sample_weights = sample_weights.detach().clone()

    @property
    def action_pairs_data(self) -> ActionPairsData:
        """Returns a clone of the stored ActionPairsData object

        Returns:
            ActionPairsData: list of action pairs
        """
        return self._action_pairs_data.clone()

    @property
    def pref_pairs_data(self) -> PreferencePairsData:
        """Returns a clone of the stored PreferencePairsData object

        Returns:
            PreferencePairsData: list of preference pairs
        """
        return self._pref_pairs_data.clone()

    @property
    def sample_weights(self) -> torch.Tensor:
        """Returns a clone of the stored per-pair sample weights tensor

        Returns:
            torch.Tensor: [B] tensor, B - batch size
        """
        return self._sample_weights.clone()

    @override
    def clone(self) -> "ActionPairsPrefPairsContainer":
        """Returns an independent deep copy of this object

        Returns:
            ActionPairsPrefPairsContainer: new object holding cloned copies
                of the underlying ActionPairsData, PreferencePairsData and
                sample weights
        """

        return ActionPairsPrefPairsContainer(
            action_pairs_data=self.action_pairs_data,
            pref_pairs_data=self.pref_pairs_data,
            sample_weights=self.sample_weights,
        )

    def __str__(self) -> str:
        """Returns a string representing an object

        Returns:
            str
        """
        return "Action Pairs Pref Pairs Container"
