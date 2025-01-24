import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

###################################################
###################################################
###################################################
###################################################


import torch

from src.DataStructures.ActionPairsPrefPairsContainer import (
    ActionPairsPrefPairsContainer,
)
from src.DataStructures.PreferencePairsData import PreferencePairsData
from src.FeedbackSource.RandomFeedbackSource import RandomFeedbackSource
from src.GenModel.StackGanGenModel import StackGanGenModel
from src.Loss.PreferenceLoss import PreferenceLoss
from src.PreferenceDataGenerator.RandomPreferenceDataGenerator import (
    RandomPreferenceDataGenerator,
)
from src.RewardModel.mlpRewardNetwork import mlpRewardNetwork


class TestLoss:
    def test_preference_cross_loss(self):
        gen_model = StackGanGenModel(
            config_file="./GenerativeModelsData/StackGan2/config/facade_3stages_color.yml",
            checkpoint_file="./GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netG_56500.pth",
            scale_level=2,
        )

        nz = gen_model.sample_random_actions(N=1).actions.shape[1]
        nh = 300

        reward_model = mlpRewardNetwork(input_dim=nz, hidden_dim=nh)

        prefLoss = PreferenceLoss(decimals=None, rewardModel=reward_model)

        feedback_source = RandomFeedbackSource()

        prefDataGenerator = RandomPreferenceDataGenerator(
            feedbackSource=feedback_source
        )

        actions = gen_model.sample_random_actions(5)

        action_pairs, preference_data = prefDataGenerator.generate_preference_data(
            data=actions, limit=4
        )

        y = torch.tensor([[1, 0], [0.5, 0.5], [1, 0], [0, 1]])

        data = ActionPairsPrefPairsContainer(
            action_pairs_data=action_pairs,
            pref_pairs_data=PreferencePairsData(preference_pairs=y),
        )

        loss = prefLoss.calculate_loss(data)

        assert isinstance(loss, torch.Tensor)
        assert list(loss.shape) == []
