import math

from torch import randperm

from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.Filter.AbsActionFilter import AbsActionFilter


class RandomActionFilter(AbsActionFilter):
    """Filter that randomly picks set amount of actions"""

    def __init__(self, limit: int | float | None):
        """

        Args:
            limit (int | float | None): maximum amount of actions filter
                can return. Can be set as absolute number of elements
                or as a percent of the original list

        Raises:
            Exception: if absolute limit value is less than 1
            Exception: if relative limit value is not in range [0,1]
        """

        """
        Params:
            rewardModel - reward model to use as estimator
            limit - max number of actions to return
        """

        self.limit = limit

        if isinstance(self.limit, int) and self.limit < 1:
            raise Exception(f"Invalid limit int value: {self.limit}")

        elif isinstance(self.limit, float) and (self.limit > 1 or self.limit < 0):
            raise Exception(f"Invalid limit float value: {self.limit}")

    def filter(self, action_data: ActionData) -> ActionData:
        """Pickes defined number of actions and returns them

        Args:
            action_data (ActionData): list of actions to filter

        Returns:
            ActionData: filtered list of actions
        """

        sorted_values = randperm(action_data.actions.shape[0])

        actions = action_data.actions[sorted_values]

        int_limit = None

        if self.limit is not None:
            if isinstance(self.limit, int):
                int_limit = self.limit
            else:
                int_limit = math.ceil(len(actions) * self.limit)

            actions = actions[-int_limit:]

        return ActionData(actions=actions)

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """
        return f"Random Action Filter. Limit: {self.limit}"
