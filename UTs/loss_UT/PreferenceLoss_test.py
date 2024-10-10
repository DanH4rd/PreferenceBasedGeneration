
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
from src.Loss.ConcreteLoss.PreferenceLoss import PreferenceLoss
from src.DataStructures.ConcreteDataStructures.PairPreferenceData import PairPreferenceData
from src.GenModel.ConcreteGenModel.StackGanGenModel import StackGanGenModel
from src.RewardModel.ConcreteRewardNetwork.mlpRewardNetwork import mlpRewardNetwork
from src.Trainer.ConcreteTrainer.ptLightningTrainer import ptLightningTrainer, ptLightningModelWrapper, ptLightningLatentWrapper
from src.Loss.ConcreteLoss.PreferenceLoss import PreferenceLoss
from src.Loss.ConcreteLoss.ActionRewardLoss import ActionRewardLoss
from src.Logger.ConcreteLogger.TensorboardScalarLogger import TensorboardScalarLogger
from src.Loss.ConcreteLoss.LogLossDecorator import LogLossDecorator
from src.FeedbackSource.ConcreteFeedbackSource.RandomFeedbackSource import RandomFeedbackSource
from src.PreferenceDataGenerator.ConcretePreferenceDataGenerator.RandomPreferenceDataGenerator import RandomPreferenceDataGenerator
from src.DataStructures.ConcreteDataStructures.ActionPairsPrefPairsContainer import ActionPairsPrefPairsContainer
from src.DataStructures.ConcreteDataStructures.ActionPairsData import ActionPairsData

gen_model = StackGanGenModel(
    config_file = './GenerativeModelsData/StackGan2/config/facade_3stages_color.yml',
    checkpoint_file = './GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netG_56500.pth',
    gen_level=2
)

nz = gen_model.GetNoise().actions.shape[1]
nh = 300

reward_model = mlpRewardNetwork(input_dim = nz, hidden_dim = nh)

prefLoss = PreferenceLoss(decimals=None, rewardModel=reward_model)

feedback_source = RandomFeedbackSource()

prefDataGenerator = RandomPreferenceDataGenerator(feedbackSource=feedback_source)

actions = gen_model.GetNoise(5)

action_pairs, preference_data = prefDataGenerator.GeneratePreferenceData(data=actions, limit=4)

y = torch.tensor([[1, 0], [0.5, 0.5], [1, 0], [0, 1]])

data = ActionPairsPrefPairsContainer(action_pairs_data=action_pairs, 
                                     pref_pairs_data=PairPreferenceData(y=y))

loss = prefLoss.CalculateLoss(data)

assert(isinstance(loss, torch.Tensor))
assert(list(loss.shape) == [])