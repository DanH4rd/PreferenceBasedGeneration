from src.RewardModel import mlpRewardNetwork


class TestRewardModel:
    def test_mlp_model(self, facade_gen_model):
        nz = facade_gen_model.sample_random_actions(N=1).actions.shape[1]
        nh = 300
        reward_model = mlpRewardNetwork(input_dim=nz, hidden_dim=nh)

        assert list(
            reward_model.get_rewards(facade_gen_model.sample_random_actions(N=1)).shape
        ) == [1, 1]
        assert list(
            reward_model.get_rewards(facade_gen_model.sample_random_actions(N=5)).shape
        ) == [5, 1]
