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

import pytest
import torch

from src.Filter.ConcreteActionFilter.RandomActionFilter import RandomActionFilter
from src.Filter.ConcreteActionFilter.ScoreActionFilter import ScoreActionFilter
from src.GenModel.ConcreteGenModel.StackGanGenModel import StackGanGenModel
from src.RewardModel.ConcreteRewardNetwork.mlpRewardNetwork import mlpRewardNetwork

########### Utils
##################################


def create_score_action_filter(mode):
    gen_model = StackGanGenModel(
        config_file="./GenerativeModelsData/StackGan2/config/facade_3stages_color.yml",
        checkpoint_file="./GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netG_56500.pth",
        gen_level=2,
    )

    actions = gen_model.sample_random_actions(N=1)

    reward_model = mlpRewardNetwork(
        input_dim=actions.actions.shape[1], hidden_dim=20, p=0
    )
    return ScoreActionFilter(
        mode=mode, key=lambda x: reward_model.get_stable_rewards(x), limit=1
    )


########### Utils End
##################################


filters = []

filters.append(RandomActionFilter(limit=1))

filters.append(create_score_action_filter(mode="max"))
filters.append(create_score_action_filter(mode="min"))


class TestFilter:

    @pytest.mark.parametrize("filter,filter_name", zip(filters, map(str, filters)))
    def test_base(self, filter, filter_name):
        gen_model = StackGanGenModel(
            config_file="./GenerativeModelsData/StackGan2/config/facade_3stages_color.yml",
            checkpoint_file="./GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netG_56500.pth",
            gen_level=2,
        )

        filter.limit = 10
        actions = gen_model.sample_random_actions(N=15)
        actions = filter.filter(actions)

        assert len(actions.actions.shape) == 2
        assert actions.actions.shape[0] == 10

        filter.limit = 0.5
        actions = gen_model.sample_random_actions(N=10)
        actions = filter.filter(actions)

        assert len(actions.actions.shape) == 2
        assert actions.actions.shape[0] == 5

    ###########

    def test_score_filter_ranking(self):

        gen_model = StackGanGenModel(
            config_file="./GenerativeModelsData/StackGan2/config/facade_3stages_color.yml",
            checkpoint_file="./GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netG_56500.pth",
            gen_level=2,
        )

        actions = gen_model.sample_random_actions(N=10)

        key = lambda x: reward_model.get_stable_rewards(x)

        reward_model = mlpRewardNetwork(
            input_dim=actions.actions.shape[1], hidden_dim=20, p=0
        )

        rewards = reward_model.get_stable_rewards(actions)

        rewards_sort = torch.argsort(rewards, dim=0).squeeze()

        sorted_actions_tensor = actions.actions[rewards_sort]

        filter = ScoreActionFilter(mode="max", key=key, limit=0.5)

        filter_actions = filter.filter(actions)

        assert ((sorted_actions_tensor[-5:] - filter_actions.actions) < 10e-5).all()

        filter = ScoreActionFilter(mode="min", key=key, limit=0.5)

        filter_actions = filter.filter(actions)

        assert ((sorted_actions_tensor[:5] - filter_actions.actions) < 10e-5).all()
