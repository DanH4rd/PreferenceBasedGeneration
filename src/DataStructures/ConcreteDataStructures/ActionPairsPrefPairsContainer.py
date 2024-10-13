from src.DataStructures.AbsData import AbsData
from src.DataStructures.ConcreteDataStructures.ActionPairsData import ActionPairsData
from src.DataStructures.ConcreteDataStructures.PreferencePairsData import (
    PreferencePairsData,
)


class ActionPairsPrefPairsContainer(AbsData):
    """
    Container class for storing ActionPairsData and PairPreferenceData together
    """

    def __init__(
        self, action_pairs_data: ActionPairsData, pref_pairs_data: PreferencePairsData
    ):

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
        return "Action Pairs Pref Pairs Container"
