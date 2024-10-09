from src.DataStructures.AbsData import AbsData
from src.DataStructures.ConcreteDataStructures.ActionPairsData import ActionPairsData
from src.DataStructures.ConcreteDataStructures.PairPreferenceData import PairPreferenceData
import torch

class ActionPairsPrefPairsContainer(AbsData):
    """
        Container class for storing ActionPairsData and PairPreferenceData together
    """

    def __init__(self, action_pairs_data:ActionPairsData, pref_pairs_data:PairPreferenceData):

        self.action_pairs_data = action_pairs_data
        self.pref_pairs_data = pref_pairs_data

    def __str__(self) -> str:
       return "Action Pairs Pref Pairs Container"