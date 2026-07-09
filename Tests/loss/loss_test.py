import torch

from src.DataStructures import (
    ActionPairsData,
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

    def test_preference_loss_sample_weights(self, facade_gen_model):
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

        action_pairs, _ = prefDataGenerator.generate_preference_data(
            data=actions, limit=4
        )

        y = torch.tensor([[1, 0], [0.5, 0.5], [1, 0], [0, 1]])

        # weighting only the first pair must make the loss equal to that
        # pair's loss computed on its own
        weighted_data = ActionPairsPrefPairsContainer(
            action_pairs_data=action_pairs,
            pref_pairs_data=PreferencePairsData(preference_pairs=y),
            sample_weights=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )
        single_pair_data = ActionPairsPrefPairsContainer(
            action_pairs_data=ActionPairsData(
                action_pairs=action_pairs.action_pairs[:1]
            ),
            pref_pairs_data=PreferencePairsData(preference_pairs=y[:1]),
        )

        weighted_loss = prefLoss.calculate_loss(weighted_data)
        single_pair_loss = prefLoss.calculate_loss(single_pair_data)

        assert torch.isclose(weighted_loss, single_pair_loss, atol=1e-6)

        # uniform (non-1.0) weights must not change the loss vs the unweighted default
        default_data = ActionPairsPrefPairsContainer(
            action_pairs_data=action_pairs,
            pref_pairs_data=PreferencePairsData(preference_pairs=y),
        )
        uniform_weights_data = ActionPairsPrefPairsContainer(
            action_pairs_data=action_pairs,
            pref_pairs_data=PreferencePairsData(preference_pairs=y),
            sample_weights=torch.full((4,), 2.0),
        )

        default_loss = prefLoss.calculate_loss(default_data)
        uniform_weighted_loss = prefLoss.calculate_loss(uniform_weights_data)

        assert torch.isclose(default_loss, uniform_weighted_loss, atol=1e-6)
