import abc
from typing import override

import torch

from src.DataStructures.ConcreteDataStructures.ActionPairsData import ActionPairsData
from src.DataStructures.ConcreteDataStructures.ActionPairsPrefPairsContainer import (
    ActionPairsPrefPairsContainer,
)
from src.DataStructures.ConcreteDataStructures.PreferencePairsData import (
    PreferencePairsData,
)


class RoundsMemory(object, metaclass=abc.ABCMeta):
    """Memory, keeping the last N number of placed data entries.
    Name comes from the base preference generation pipeline,
    where data to memory is added at the end of each round
    """

    def __init__(self, limit: int, discount_factor: float | None = None) -> None:
        """
        Args:
            limit (int) - number of last data entries to keep
            discount_factor (float|None, optional): if float (0<x<=1) will apply a multiplier to
                          preference labels equal to `discount_factor^n` when extracting data
                          from memory, where n is the position of data entry.
                          Defaults to None.

        Raises:
            Exception: if the discount factor is float and is not in range (0,1]
            Exception: if the discount factor is neither None nor float
        """

        self.memory_list = []
        self.limit = limit
        self.discount_factor = discount_factor

        if self.discount_factor is not None:
            if isinstance(self.discount_factor, float):
                if self.discount_factor <= 0:
                    raise Exception(
                        f"Discount factor cannot be lower than 0, received: {self.discount_factor}"
                    )
            else:
                raise Exception(
                    f"Wrong type of discount factor ({type(self.discount_factor)}), expected float"
                )
        pass

    @override
    def add_data(self, data: ActionPairsPrefPairsContainer) -> None:
        """Add new preference and action data to the memory and remove
        old data entries

        Args:
            data (ActionPairsPrefPairsContainer): action pair list and corrensonding preferences
                to add to memory
        """

        self.memory_list.append(data)
        self.memory_list = self.memory_list[-self.limit :]

    @override
    def get_data_from_memory(self) -> ActionPairsPrefPairsContainer:
        """Returns the data kept in memory with discount factor multiplier
        applied if set.

        Returns:
            ActionPairsPrefPairsContainer: data from memory
        """

        action_pairs_list = []
        pref_pairs_list = []

        memory_length = len(self.memory_list)

        # conbine action pairs lists and preference lists from all
        # kept container objects into one action pair list and preference list
        # and apply the discount factor multiplier if set
        for i, data in enumerate(self.memory_list):
            action_pairs_list.append(data.action_pairs_data.action_pairs)
            pref_tensor_entry = data.pref_pairs_data.preference_pairs

            if self.discount_factor is not None:
                pref_tensor_entry *= pow(self.discount_factor, (memory_length - i))

            pref_pairs_list.append(pref_tensor_entry)

        action_pairs_tensor = torch.concat(action_pairs_list, dim=0)
        pref_pairs_tensor = torch.concat(pref_pairs_list, dim=0)

        # pack memory data in a corresponding class object
        action_pairs_data = ActionPairsData(action_pairs=action_pairs_tensor)
        pref_pairs_data = PreferencePairsData(preference_pairs=pref_pairs_tensor)

        pref_action_container = ActionPairsPrefPairsContainer(
            action_pairs_data=action_pairs_data, pref_pairs_data=pref_pairs_data
        )

        return pref_action_container

    @override
    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """

        return (
            "Round Memory Object" + ""
            if self.discount_factor is None
            else f" with discount ({self.discount_factor})"
        )
