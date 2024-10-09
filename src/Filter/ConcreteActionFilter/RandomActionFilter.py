from src.Filter.AbsActionFilter import AbsActionFilter
from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from torch import randperm
import math

class RandomActionFilter(AbsActionFilter):
    """
        Filter that returns set amount of action picked randomly
    """

    def __init__(self, limit:int|float|None):
        """
            Params:
                rewardModel - reward model to use as estimator 
                limit - max number of actions to return
        """

        self.limit = limit

        if self.limit is not None:
            if isinstance(self.limit, int):
                if self.limit < 1:
                    raise Exception(f'Invalid limit int value: {self.limit}')
            elif isinstance(self.limit, float):
                if self.limit > 1 or self.limit < 0:
                    raise Exception(f'Invalid limit float value: {self.limit}')
            else:
                raise Exception(f'Wrong limit value type: {self.limit} - {type(self.limit)}')



    def Filter(self, data:ActionData) -> ActionData:
        """
            Pickes limit of random actions and returns them

            Check the abstract base class for more info.
        """

        sorted_values = randperm(data.actions.shape[0])

        actions = data.actions[sorted_values]

        int_limit = None

        if self.limit is not None:
            if isinstance(self.limit, int):
                int_limit = self.limit
            else:
                int_limit = math.ceil(len(actions) * self.limit)

            actions = actions[-int_limit:]

        return ActionData(actions=actions)

    def __str__(self) -> str:
        return f"Random Action Filter. Limit: {len(self.limit)}"