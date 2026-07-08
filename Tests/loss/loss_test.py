import torch

from src.DataStructures import (
    ActionPairsPrefPairsContainer,
    PreferencePairsData,
)
from src.FeedbackSource import RandomFeedbackSource
from src.Loss import PreferenceLoss
from src.PreferenceDataGenerator import (
    RandomPreferenceDataGenerator,
)
from src.RewardModel import mlpRewardNetwork


class TestLoss:
    def test_preference_cross_loss(self, facade_gen_model):
        gen_model = facade_gen_model

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
