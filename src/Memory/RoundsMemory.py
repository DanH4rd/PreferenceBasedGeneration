import abc
from dataclasses import dataclass
from typing import override

import torch

from src.DataStructures import (
    ActionPairsData,
    ActionPairsPrefPairsContainer,
    PreferencePairsData,
)


class RoundsMemory(object, metaclass=abc.ABCMeta):
    """Memory, keeping the last N number of placed data entries.
    Name comes from the base preference generation pipeline,
    where data to memory is added at the end of each round
    """

    @dataclass
    class Configuration:
        """dataclass for grouping constructor parametres"""

        limit: int
        discount_factor: float | None = None

    @staticmethod
    def create_from_configuration(conf: Configuration):
        return RoundsMemory(limit=conf.limit, discount_factor=conf.discount_factor)

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

    def add_data(self, data: ActionPairsPrefPairsContainer) -> None:
        """Add new preference and action data to the memory and remove
        old data entries

        Args:
            data (ActionPairsPrefPairsContainer): action pair list and corrensonding preferences
                to add to memory
        """

        self.memory_list.append(data)
        self.memory_list = self.memory_list[-self.limit :]

    def get_data_from_memory(self) -> ActionPairsPrefPairsContainer:
        """Returns the data kept in memory, with a per-pair sample weight
        reflecting the discount factor if set (1.0 for every pair otherwise).

        The discount is carried as a sample weight rather than baked into the
        preference values themselves: multiplying a preference pair (e.g.
        [1., 0.]) by a decaying factor breaks PreferencePairsData's legal-value
        invariant (must be non-negative and sum to 1, or be the [0., 0.] skip
        sentinel), and conflates "how confident is this preference" with "how
        much should this round count towards the loss".

        Returns:
            ActionPairsPrefPairsContainer: data from memory
        """

        action_pairs_list = []
        pref_pairs_list = []
        weights_list = []

        memory_length = len(self.memory_list)

        # conbine action pairs lists and preference lists from all
        # kept container objects into one action pair list and preference list,
        # weighting each round's pairs by the discount factor if set
        for i, data in enumerate(self.memory_list):
            action_pairs_list.append(data.action_pairs_data.action_pairs)
            pref_tensor_entry = data.pref_pairs_data.preference_pairs
            pref_pairs_list.append(pref_tensor_entry)

            entry_weight = (
                pow(self.discount_factor, (memory_length - i))
                if self.discount_factor is not None
                else 1.0
            )
            weights_list.append(torch.full((pref_tensor_entry.shape[0],), entry_weight))

        action_pairs_tensor = torch.concat(action_pairs_list, dim=0)
        pref_pairs_tensor = torch.concat(pref_pairs_list, dim=0)
        sample_weights = torch.concat(weights_list, dim=0)

        # pack memory data in a corresponding class object
        action_pairs_data = ActionPairsData(action_pairs=action_pairs_tensor)
        pref_pairs_data = PreferencePairsData(preference_pairs=pref_pairs_tensor)

        pref_action_container = ActionPairsPrefPairsContainer(
            action_pairs_data=action_pairs_data,
            pref_pairs_data=pref_pairs_data,
            sample_weights=sample_weights,
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
