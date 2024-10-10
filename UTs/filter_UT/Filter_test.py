
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir) 

###################################################
###################################################
###################################################
###################################################

import torch
from src.GenModel.ConcreteGenModel.StackGanGenModel import StackGanGenModel
from src.Filter.ConcreteActionFilter.RandomActionFilter import RandomActionFilter
from src.Filter.ConcreteActionFilter.ScoreActionFilter import ScoreActionFilter
from src.RewardModel.ConcreteRewardNetwork.mlpRewardNetwork import mlpRewardNetwork

gen_model = StackGanGenModel(
    config_file = './GenerativeModelsData/StackGan2/config/facade_3stages_color.yml',
    checkpoint_file = './GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netG_56500.pth',
    gen_level=2
)

filter = RandomActionFilter(limit = 10)
actions = gen_model.GetNoise(15)
actions = filter.Filter(actions)

assert(len(actions.actions.shape) == 2)
assert(actions.actions.shape[0] == 10)


filter = RandomActionFilter(limit = 0.5)
actions = gen_model.GetNoise(10)
actions = filter.Filter(actions)

assert(len(actions.actions.shape) == 2)
assert(actions.actions.shape[0] == 5)

###########


reward_model = mlpRewardNetwork(input_dim=actions.actions.shape[1], hidden_dim=20, p = 0)

reward_model.SetToEvaluaionMode()

key = lambda x: reward_model.GetStableRewards(x)

filter = ScoreActionFilter(mode='max', key = key, limit=10)
actions = gen_model.GetNoise(15)
actions = filter.Filter(actions)
assert(len(actions.actions.shape) == 2)
assert(actions.actions.shape[0] == 10)


filter = ScoreActionFilter(mode='max', key = key, limit=0.5 )
actions = gen_model.GetNoise(10)
actions = filter.Filter(actions)
assert(len(actions.actions.shape) == 2)
assert(actions.actions.shape[0] == 5)


actions = gen_model.GetNoise(10)

rewards = reward_model.GetRewards(actions)

rewards_sort = torch.argsort(rewards, dim=0).squeeze()

sorted_actions_tensor = actions.actions[rewards_sort]


filter = ScoreActionFilter(mode='max', key = key, limit=0.5)

filter_actions = filter.Filter(actions)

assert(((sorted_actions_tensor[-5:] - filter_actions.actions) < 10e-5).all())


filter = ScoreActionFilter(mode='min', key = key, limit=0.5)

filter_actions = filter.Filter(actions)

assert(((sorted_actions_tensor[:5] - filter_actions.actions) < 10e-5).all())

