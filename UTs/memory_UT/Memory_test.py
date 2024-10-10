
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

from src.PreferenceDataGenerator.ConcretePreferenceDataGenerator.RandomPreferenceDataGenerator import RandomPreferenceDataGenerator
from src.GenModel.ConcreteGenModel.StackGanGenModel import StackGanGenModel
from src.RewardModel.ConcreteRewardNetwork.mlpRewardNetwork import mlpRewardNetwork
from src.Trainer.ConcreteTrainer.ptLightningTrainer import ptLightningTrainer, ptLightningModelWrapper, ptLightningLatentWrapper
from src.Loss.ConcreteLoss.PreferenceLoss import PreferenceLoss
from src.Loss.ConcreteLoss.ActionRewardLoss import ActionRewardLoss
from src.Logger.ConcreteLogger.TensorboardScalarLogger import TensorboardScalarLogger
from src.Loss.ConcreteLoss.LogLossDecorator import LogLossDecorator
from src.DataStructures.ConcreteDataStructures.ActionPairsPrefPairsContainer import ActionPairsPrefPairsContainer
from src.FeedbackSource.ConcreteFeedbackSource.RandomFeedbackSource import RandomFeedbackSource
from src.PreferenceDataGenerator.ConcretePreferenceDataGenerator.RandomPreferenceDataGenerator import RandomPreferenceDataGenerator
from src.Memory.ConcreteMemory.RoundsMemory import RoundsMemory

gen_model = StackGanGenModel(
    config_file = './GenerativeModelsData/StackGan2/config/facade_3stages_color.yml',
    checkpoint_file = './GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netG_56500.pth',
    gen_level=2
)
feedback_source = RandomFeedbackSource()

prefDataGenerator = RandomPreferenceDataGenerator(feedbackSource=feedback_source)

actions = gen_model.GetNoise(5)

action_data, pref_data = prefDataGenerator.GeneratePreferenceData(data=actions, limit = 8)

action_pref_data = ActionPairsPrefPairsContainer(action_pairs_data=action_data, pref_pairs_data=pref_data)

memory = RoundsMemory(limit=3)

memory.AddData(action_pref_data)

data = memory.GetMemoryData()

assert(len(memory.memory_list) == 1)
assert(data.action_pairs_data.actions_pairs.shape[0] == 8)
assert(data.pref_pairs_data.y.shape[0] == 8)


memory.AddData(action_pref_data)
memory.AddData(action_pref_data)
memory.AddData(action_pref_data)
memory.AddData(action_pref_data)

data = memory.GetMemoryData()

assert(len(memory.memory_list) == 3)
assert(data.action_pairs_data.actions_pairs.shape[0] == 24)
assert(data.pref_pairs_data.y.shape[0] == 24)
