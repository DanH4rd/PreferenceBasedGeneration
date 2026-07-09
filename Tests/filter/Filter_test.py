import pytest
import torch

from src.Filter import RandomActionFilter, ScoreActionFilter
from src.RewardModel import mlpRewardNetwork


def _score_action_filter(gen_model, mode):
    actions = gen_model.sample_random_actions(N=1)
    reward_model = mlpRewardNetwork(
        input_dim=actions.actions.shape[1], hidden_dim=20, p=0
    )
    return ScoreActionFilter(
        mode=mode, key=lambda x: reward_model.get_stable_rewards(x), limit=1
    )


FILTER_FACTORIES = {
    "random": lambda gen_model: RandomActionFilter(limit=1),
    "score_max": lambda gen_model: _score_action_filter(gen_model, mode="max"),
    "score_min": lambda gen_model: _score_action_filter(gen_model, mode="min"),
}


class TestFilter:
    @pytest.fixture(params=list(FILTER_FACTORIES))
    def action_filter(self, request, facade_gen_model):
        return FILTER_FACTORIES[request.param](facade_gen_model)

    def test_base(self, action_filter, facade_gen_model):
        action_filter.limit = 10
        actions = facade_gen_model.sample_random_actions(N=15)
        actions = action_filter.filter(actions)

        assert len(actions.actions.shape) == 2
        assert actions.actions.shape[0] == 10

        action_filter.limit = 0.5
        actions = facade_gen_model.sample_random_actions(N=10)
        actions = action_filter.filter(actions)

        assert len(actions.actions.shape) == 2
        assert actions.actions.shape[0] == 5

    ###########

    def test_score_filter_ranking(self, facade_gen_model):
        actions = facade_gen_model.sample_random_actions(N=10)

        reward_model = mlpRewardNetwork(
            input_dim=actions.actions.shape[1], hidden_dim=20, p=0
        )

        def key(x):
            return reward_model.get_stable_rewards(x)

        rewards = reward_model.get_stable_rewards(actions).flatten()

        ranking_desc = torch.argsort(rewards, descending=True)

        # filtered output preserves the actions' original relative arrangement,
        # so the expected subset must be re-sorted back into original index order
        score_filter = ScoreActionFilter(mode="max", key=key, limit=0.5)

        filter_actions = score_filter.filter(actions)

        expected_idx = torch.sort(ranking_desc[:5]).values

        assert ((actions.actions[expected_idx] - filter_actions.actions) < 10e-5).all()

        score_filter = ScoreActionFilter(mode="min", key=key, limit=0.5)

        filter_actions = score_filter.filter(actions)

        expected_idx = torch.sort(ranking_desc[-5:]).values

        assert ((actions.actions[expected_idx] - filter_actions.actions) < 10e-5).all()
