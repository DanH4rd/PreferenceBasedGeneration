from typing import override

from src.Abstract.AbsData import AbsData

from .ActionPairsData import ActionPairsData
from .PreferencePairsData import PreferencePairsData


class ActionPairsPrefPairsContainer(AbsData):
    """Class for storing ActionPairsData and PreferencepairData together"""

    def __init__(
        self, action_pairs_data: ActionPairsData, pref_pairs_data: PreferencePairsData
    ):
        """

        Args:
            action_pairs_data (ActionPairsData): list of action pairs
            pref_pairs_data (PreferencePairsData): list of preference pairs

        Raises:
            Exception: if action_pairs_data doesn't have the same number
                of elements (value of the zero dim size) as pref_pairs_data
        """

        if (
            action_pairs_data.action_pairs.shape[0]
            != pref_pairs_data.preference_pairs.shape[0]
        ):
            raise Exception(
                f"Action pairs number does not correspond to preference pairs number: {action_pairs_data.action_pairs.shape[0]} and {pref_pairs_data.preference_pairs.shape[0]}"
            )

        self._action_pairs_data = action_pairs_data.clone()
        self._pref_pairs_data = pref_pairs_data.clone()

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

    @override
    def clone(self) -> "ActionPairsPrefPairsContainer":
        """Returns an independent deep copy of this object

        Returns:
            ActionPairsPrefPairsContainer: new object holding cloned copies
                of the underlying ActionPairsData and PreferencePairsData
        """

        return ActionPairsPrefPairsContainer(
            action_pairs_data=self.action_pairs_data,
            pref_pairs_data=self.pref_pairs_data,
        )

    def __str__(self) -> str:
        """Returns a string representing an object

        Returns:
            str
        """
        return "Action Pairs Pref Pairs Container"
