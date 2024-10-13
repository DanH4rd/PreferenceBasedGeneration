import math
from collections.abc import Callable

import torch

from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.Filter.AbsActionFilter import AbsActionFilter


class ScoreActionFilter(AbsActionFilter):
    """
    Filter that returns set amount of action based on the provided scoring

    !!! Returns actions sorted in an ascending way based on their scores !!!
    """

    def __init__(
        self,
        mode: str,
        key: Callable[[ActionData], torch.tensor],
        limit: int | float | None,
    ):
        """
        Params:
            key - lambda function, accepting ActionData as parametre and returning torch.tensor of scores
                    for each action
            limit - max number of actions to return
        """

        self.key = key
        self.limit = limit
        self.mode = mode

        if self.limit is not None:
            if isinstance(self.limit, int):
                if self.limit < 1:
                    raise Exception(f"Invalid limit int value: {self.limit}")
            elif isinstance(self.limit, float):
                if self.limit > 1 or self.limit < 0:
                    raise Exception(f"Invalid limit float value: {self.limit}")
            else:
                raise Exception(
                    f"Wrong limit value type: {self.limit} - {type(self.limit)}"
                )

    def Filter(self, data: ActionData) -> ActionData:
        """
        Calculates scores for actions provided and returns
        limit of actions with the biggest score

        Check the abstract base class for more info.
        """

        scores = self.key(data)

        sorted_values = torch.argsort(scores, dim=0).squeeze()

        actions = data.actions[sorted_values]

        int_limit = None

        if self.limit is not None:
            if isinstance(self.limit, int):
                int_limit = self.limit
            else:
                int_limit = math.ceil(len(actions) * self.limit)

            if self.mode == "max":
                actions = actions[-int_limit:]
            elif self.mode == "min":
                actions = actions[:int_limit]
            else:
                raise Exception(f"Invalid filter mode for ({str(self)}): {self.mode}")

        return ActionData(actions=actions)

    def __str__(self) -> str:
        return f"Reward Action Filter. Mode: {self.mode}. Limit: {len(self.limit)}"
