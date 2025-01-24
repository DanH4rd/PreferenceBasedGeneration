from src.Abstract.AbsData import AbsData
from src.DataStructures.ActionPairsData import ActionPairsData
from src.DataStructures.PreferencePairsData import PreferencePairsData


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

        self.action_pairs_data = action_pairs_data
        self.pref_pairs_data = pref_pairs_data

        if (
            self.action_pairs_data.action_pairs.shape[0]
            != self.pref_pairs_data.preference_pairs.shape[0]
        ):
            raise Exception(
                f"Action pairs number does not correspond to preference pairs number: {self.action_pairs_data.actions_pairs.shape[0]} and {self.pref_pairs_data.y.shape[0]}"
            )

    def __str__(self) -> str:
        """Returns a string representing an object

        Returns:
            str
        """
        return "Action Pairs Pref Pairs Container"
