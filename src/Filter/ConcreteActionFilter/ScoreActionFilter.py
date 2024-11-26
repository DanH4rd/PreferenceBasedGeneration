import math
from collections.abc import Callable

import torch

from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.Filter.AbsActionFilter import AbsActionFilter


class ScoreActionFilter(AbsActionFilter):
    """Filter that returns set amount of action based on the provided scoring function

    !!! Returns actions sorted in an descending way based on their scores !!!
    """

    def __init__(
        self,
        mode: str,
        key: Callable[[ActionData], torch.tensor],
        limit: int | float | None,
    ):
        """

        Args:
            mode (str): operating mode of the filter:
                'max' - returns actions with the largest score values;
                'min' - returns actions with the lowest score values;
            key (Callable[[ActionData], torch.tensor]): lambda function that defines the
                score we base the filtering on
            limit (int | float | None): maximum amount of actions filter
                can return. Can be set as absolute number of elements
                or as a percent of the original list

        Raises:
            Exception: if absolute limit value is less than 1
            Exception: if relative limit value is not in range [0,1]

        TODO:
            preserve original arrangement
        """

        self.key = key
        self.limit = limit
        self.mode = mode

        if isinstance(self.limit, int) and self.limit < 1:
            raise Exception(f"Invalid limit int value: {self.limit}")

        elif isinstance(self.limit, float) and (self.limit > 1 or self.limit < 0):
            raise Exception(f"Invalid limit float value: {self.limit}")

    def filter(self, action_data: ActionData) -> ActionData:
        """Calculates score for each action and returns
        actions based on them according to operating mode.

        Args:
            action_data (ActionData): _description_

        Raises:
            Exception: _description_

        Returns:
            ActionData: _description_
        """

        scores = self.key(action_data)

        sorted_values = torch.argsort(scores, dim=0).squeeze()

        actions = action_data.actions[sorted_values]

        int_limit = None

        if self.limit is not None:
            if isinstance(self.limit, int):
                int_limit = self.limit
            else:
                int_limit = math.ceil(len(actions) * self.limit)

            if self.mode == "max":
                actions = actions[-int_limit:]
                actions = torch.flip(actions, dims=[0])
            elif self.mode == "min":
                actions = actions[:int_limit]
                actions = torch.flip(actions, dims=[0])
            else:
                raise Exception(f"Invalid filter mode for ({str(self)}): {self.mode}")

        return ActionData(actions=actions)

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """
        return f"Score Action Filter. Mode: {self.mode}. Limit: {self.limit}"
