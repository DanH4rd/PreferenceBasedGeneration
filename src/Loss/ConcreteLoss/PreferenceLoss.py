import abc
import torch
from src.Loss.AbsLoss import AbsLoss
from src.DataStructures.ConcreteDataStructures.ActionPairsPrefPairsContainer import ActionPairsPrefPairsContainer
from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.RewardModel.AbsRewardModel import AbsRewardModel

class PreferenceLoss(AbsLoss):
    """
        Class than calculates preference loss. If provided decimals will round the estimated
        preference values
    """

    def __init__(self, rewardModel:AbsRewardModel, decimals:None|int = None):
        """
            Parametres:
                rewardModel - model to get estimated rewards with
                decimals - if not None will round preference values to given decimals number
        """

        self.decimals = decimals
        self.rewardModel = rewardModel

        if self.decimals is not None:
            if self.decimals < 1 :
                raise Exception(f'Invalid decimals value: {self.decimals}')

    def ConvertRewards2Preferences(self, r1:torch.tensor, r2:torch.tensor):
        """
            Function that converts reward values to preference probability

            Parametres:
                r1 - [B,1] tensor of 1st elemets of pairs, B - number of pairs (batch size)
                r2 - [B,1] tensor of 2nd elemets of pairs, B - number of pairs (batch size)

            Returns:
                tensor containing preference probability for pairs
        """

        answer = torch.exp(r1)/(torch.exp(r1) + torch.exp(r2))
        return answer

    def CalculateLoss(self, data:ActionPairsPrefPairsContainer) -> torch.tensor:
        """
            Calculates Cross Entropy loss on preference data

            Parametres:
                data - ActionPairsData and PrefPairsData Container

            ToDo:
                Ensure estimated preferences probs sum to 1 if rounding is enabled

        """

        action_pairs_tensor = data.action_pairs_data.actions_pairs
        pref_pairs_tensor = data.pref_pairs_data.y

        returnToTrainMode = False
        if self.rewardModel.IsTrainMode():
            self.rewardModel.SetToEvaluaionMode()
            returnToTrainMode = True

        rewards_left_column = self.rewardModel(ActionData(actions=action_pairs_tensor[:,0,:])).squeeze(1)
        rewards_right_column = self.rewardModel(ActionData(actions=action_pairs_tensor[:,1,:])).squeeze(1)

        if returnToTrainMode:
            self.rewardModel.SetToTrainMode()

        preferences_left_column  = self.ConvertRewards2Preferences(rewards_left_column, rewards_right_column)
        preferences_right_column = self.ConvertRewards2Preferences(rewards_right_column, rewards_left_column)

        if self.decimals is not None:
            preferences_left_column = torch.round(preferences_left_column, decimals=self.decimals)
            preferences_right_column = torch.round(preferences_right_column, decimals=self.decimals)

        loss = ( pref_pairs_tensor[...,0] * torch.log(preferences_left_column) + 
                 pref_pairs_tensor[...,1] *  torch.log(preferences_right_column))

        loss = - loss.mean()
            
        return loss

    def __str__(self) -> str:
        """
            Returns string describing the object
        """
        return f"Preference loss. Round to decimals: {self.decimals}"