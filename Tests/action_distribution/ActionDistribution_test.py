from src.ActionDistribution import (
    GreedyNormalActionDistribution,
    SimpleActionDistribution,
)
from src.DataStructures import ActionData, TrainableActionData


class TestActionDistribution:
    def test_basic(self, facade_gen_model):
        dist = SimpleActionDistribution(
            dist=facade_gen_model.get_input_noise_distribution()
        )

        assert isinstance(dist.sample(N=1), ActionData)
        assert dist.sample(N=1).actions.shape[0] == 1
        assert dist.sample(N=4).actions.shape[0] == 4

    def test_greedy(self, facade_gen_model):
        e_start = 0.9
        decay_val = 0.8

        dist = GreedyNormalActionDistribution(
            dist=facade_gen_model.get_input_noise_distribution(),
            destination_action_trainable=TrainableActionData.from_action_data(
                facade_gen_model.sample_random_actions(N=1)
            ),
            e=e_start,
            decay_factor=decay_val,
            omega2=0.5,
        )

        assert isinstance(dist.sample(N=1), ActionData)
        assert dist.sample(N=1).actions.shape[0] == 1
        assert dist.sample(N=4).actions.shape[0] == 4

        dist.update(None)
        dist.update(None)
        dist.update(None)
        dist.update(None)
        dist.update(None)
        dist.update(None)

        assert (
            dist.e
            == e_start
            * decay_val
            * decay_val
            * decay_val
            * decay_val
            * decay_val
            * decay_val
        )

        assert dist.sample(N=1).actions.shape[0] == 1
        assert dist.sample(N=4).actions.shape[0] == 4
