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


from src.FeedbackSource.ConcreteFeedbackSource.RandomFeedbackSource import (
    RandomFeedbackSource,
)
from src.GenModel.ConcreteGenModel.StackGanGenModel import StackGanGenModel
from src.PreferenceDataGenerator.ConcretePreferenceDataGenerator.BestActionTracker import (
    BestActionTracker,
)
from src.PreferenceDataGenerator.ConcretePreferenceDataGenerator.GraphPreferenceDataGeneration import (
    GraphPreferenceDataGeneration,
)
from src.PreferenceDataGenerator.ConcretePreferenceDataGenerator.RandomPreferenceDataGenerator import (
    RandomPreferenceDataGenerator,
)

gen_model = StackGanGenModel(
    config_file="./GenerativeModelsData/StackGan2/config/facade_3stages_color.yml",
    checkpoint_file="./GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netG_56500.pth",
    gen_level=2,
)

feedback_source = RandomFeedbackSource()

prefDataGenerator = RandomPreferenceDataGenerator(feedbackSource=feedback_source)

actions = gen_model.GetNoise(5)

action_pairs, preference_data = prefDataGenerator.GeneratePreferenceData(
    data=actions, limit=10
)

assert list(action_pairs.actions_pairs.shape) == [10, 2, actions.actions.shape[1]]
assert list(preference_data.y.shape) == [10, 2]


actions = gen_model.GetNoise(45)

action_pairs, preference_data = prefDataGenerator.GeneratePreferenceData(
    data=actions, limit=10
)

assert list(action_pairs.actions_pairs.shape) == [10, 2, actions.actions.shape[1]]
assert list(preference_data.y.shape) == [10, 2]

###########


prefDataGenerator = GraphPreferenceDataGeneration(feedbackSource=feedback_source)

actions = gen_model.GetNoise(5)

action_pairs, preference_data = prefDataGenerator.GeneratePreferenceData(
    data=actions, limit=99
)

assert list(action_pairs.actions_pairs.shape) == [10, 2, actions.actions.shape[1]]


assert list(preference_data.y.shape) == [10, 2]

actions = gen_model.GetNoise(45)

action_pairs, preference_data = prefDataGenerator.GeneratePreferenceData(
    data=actions, limit=10
)

assert list(action_pairs.actions_pairs.shape) == [10, 2, actions.actions.shape[1]]
assert list(preference_data.y.shape) == [10, 2]


###########

prefDataGenerator = BestActionTracker(prefDataGen=prefDataGenerator)


actions = gen_model.GetNoise(15)

action_pairs, preference_data = prefDataGenerator.GeneratePreferenceData(
    data=actions, limit=10
)

# Sometimes BestActionTraker may lose some actions in the final stage of additional data
# generation, check todo in BestActionTracker implementation
assert list(action_pairs.actions_pairs.shape) == [
    24,
    2,
    actions.actions.shape[1],
]  # 10 usual + 14 aditional pairs
# (best + each other action)
assert list(preference_data.y.shape) == [24, 2]
