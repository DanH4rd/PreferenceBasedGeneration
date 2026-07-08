import enum
import math
from collections.abc import Callable

import torch

from src.Abstract.AbsActionFilter import AbsActionFilter
from src.DataStructures import ActionData


class ScoreActionFilter(AbsActionFilter):
    """Filter that returns set amount of action based on the provided scoring function

    !!! Returns actions sorted in an descending way based on their scores !!!
    """

    class FilterMode(enum.Enum):
        """Enum class for filter operating modes"""

        MAX = "max"
        MIN = "min"

    def __init__(
        self,
        mode: str | FilterMode,
        key: Callable[[ActionData], torch.Tensor],
        limit: int | float | None,
    ):
        """

        Args:
            mode (str | FilterMode): operating mode of the filter:
                'max' - returns actions with the largest score values;
                'min' - returns actions with the lowest score values;
            key (Callable[[ActionData], torch.Tensor]): lambda function that defines the
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

        if not isinstance(self.mode, self.FilterMode):
            try:
                self.mode = self.FilterMode(self.mode)
            except ValueError as e:
                raise Exception(
                    f"Invalid filter mode for ({str(self)}): {self.mode}. Expected one of {[e.value for e in self.FilterMode]}"
                ) from e

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

        sorted_values_desc = torch.argsort(scores, dim=0, descending=True).squeeze()

        actions_desc = action_data.actions[sorted_values_desc]

        int_limit = None

        if self.limit is not None:
            if isinstance(self.limit, int):
                int_limit = self.limit
            else:
                int_limit = math.ceil(len(actions_desc) * self.limit)

            match self.mode:
                case self.FilterMode.MAX:
                    actions_desc = actions_desc[:int_limit]
                case self.FilterMode.MIN:
                    actions_desc = (
                        actions_desc[-int_limit:] if int_limit > 0 else actions_desc[:0]
                    )
                case _:
                    raise Exception(f"Filter mode not implemented: {self.mode}")

        return ActionData(actions=actions_desc)

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """
        mode_str = (
            self.mode.value if isinstance(self.mode, self.FilterMode) else self.mode
        )
        return f"Score Action Filter. Mode: {mode_str}. Limit: {self.limit}"
