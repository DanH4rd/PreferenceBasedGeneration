import abc
from typing import override
import torch
from src.DataStructures.ConcreteDataStructures.ActionPairsPrefPairsContainer import ActionPairsPrefPairsContainer
from src.DataStructures.ConcreteDataStructures.ActionPairsData import ActionPairsData
from src.DataStructures.ConcreteDataStructures.PairPreferenceData import PairPreferenceData

class RoundsMemory(object, metaclass=abc.ABCMeta):
    """
        Memory, keeping data from the last N added data entries.
        Name comes from base pipeline, when data to memory
        is added at the end of each round

        Parametres:
            limit - number of last data entries to keep
            discount_factor - if float (0<x<=1) will apply a modifier to
                              preference labels equal to discount_factor^n,
                              where n is the position of data entry in
                              memory (from newwest to oldest)
    """

    def __init__(self, limit, discount_factor = None) -> None:

        self.memory_list = []
        self.limit = limit
        self.discount_factor = discount_factor

        if self.discount_factor is not None:
            if isinstance(self.discount_factor, float):
                if self.discount_factor <= 0:
                    raise Exception(f'Discount factor cannot be lower than 0, received: {self.discount_factor}')
            else:
                raise Exception(f'Wrong type of discount factor ({type(self.discount_factor)}), expected float')
        pass


    @override
    def AddData(self, data:ActionPairsPrefPairsContainer) -> None:
        """
            Add new preference and action data to the memory
            Params:
                data - ActionPairsPrefPairsContainer to add
        """

        self.memory_list.append(data)
        self.memory_list = self.memory_list[-self.limit:]

    @override
    def GetMemoryData(self) -> ActionPairsPrefPairsContainer:
        """
            Returns data, contained in memory

            Returns:
                AbsData object
        """

        action_pairs_list = []
        pref_pairs_list = []

        memory_length = len(self.memory_list)

        for i, data in enumerate(self.memory_list):
           action_pairs_list.append(data.action_pairs_data.actions_pairs)
           pref_tensor_entry = data.pref_pairs_data.y
            
           if self.discount_factor is not None:
            pref_tensor_entry *= self.discount_factor^(memory_length - i)

           pref_pairs_list.append(pref_tensor_entry)

        action_pairs_tensor = torch.concat(action_pairs_list, dim=0)
        pref_pairs_tensor = torch.concat(pref_pairs_list, dim=0)

        action_pairs_data = ActionPairsData(actions_pairs=action_pairs_tensor)
        pref_pairs_data = PairPreferenceData(y=pref_pairs_tensor)

        pref_action_container = ActionPairsPrefPairsContainer(action_pairs_data=action_pairs_data, pref_pairs_data=pref_pairs_data)

        return pref_action_container

    @override
    def __str__(self) -> str:
        """
            Returns string describing the object
        """
        return 'Round Memory Object' + '' if self.discount_factor is None else f' with discount ({self.discount_factor})'