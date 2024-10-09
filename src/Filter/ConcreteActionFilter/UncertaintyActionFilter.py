from Filter.AbsActionFilter import AbsActionFilter
from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from RewardModel.AbsRewardModel import AbsRewardModel
import torch
import math

class UncertaintyActionFilter(AbsActionFilter):
    """
        Filter that returns set amount of action based on their uncertainty. RewardModel
        must have a NetworkExtension that hold implementation for uncertainty calculation

        !!! Returns actions sorted in an ascending way based on their uncertainty score !!!
    """

    def __init__(self, mode:str, rewardModel:AbsRewardModel, limit:int|float|None):
        """
            Params:
                rewardModel - reward model to use as estimator 
                limit - max number of actions to return
                mode - determines is filter returns based on min values or max values
        """

        self.rewardModel = rewardModel
        self.limit = limit
        self.mode = mode

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
            Calculates uncertainty score for actions provided and returns
            limit of actions with the biggest uncertainty scores

            Check the abstract base class for more info.
        """

        modelExtension = self.rewardModel.GetExtension()

        uncertainty_scores = modelExtension.CallExtensionMethod('CalculateUncertainty',[data.actions])

        sorted_values = torch.argsort(uncertainty_scores, dim = 0).squeeze()

        actions = data.actions[sorted_values]

        int_limit = None

        if self.limit is not None:
            if isinstance(self.limit, int):
                int_limit = self.limit
            else:
                int_limit = math.ceil(len(actions) * self.limit)

            if self.mode == 'max':
                actions = actions[-int_limit:]
            elif self.mode == 'min':
                actions = actions[:int_limit]
            else:
                raise Exception(f'Invalid filter mode for ({str(self)}): {self.mode}')


        return ActionData(actions=actions)

    def __str__(self) -> str:
        return f"Uncertainty Action Filter. Mode: {self.mode}. Limit: {len(self.limit)}"