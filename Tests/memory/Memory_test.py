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

from src.DataStructures.ActionPairsPrefPairsContainer import (
    ActionPairsPrefPairsContainer,
)
from src.FeedbackSource.RandomFeedbackSource import RandomFeedbackSource
from src.GenModel.StackGanGenModel import StackGanGenModel
from src.Memory.RoundsMemory import RoundsMemory
from src.PreferenceDataGenerator.RandomPreferenceDataGenerator import (
    RandomPreferenceDataGenerator,
)


class TestMemory:

    def test_rounds_memory(self):
        gen_model = StackGanGenModel(
            config_file="./GenerativeModelsData/StackGan2/config/facade_3stages_color.yml",
            checkpoint_file="./GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netG_56500.pth",
            scale_level=2,
        )
        feedback_source = RandomFeedbackSource()

        prefDataGenerator = RandomPreferenceDataGenerator(
            feedbackSource=feedback_source
        )

        actions = gen_model.sample_random_actions(N=5)

        action_data, pref_data = prefDataGenerator.generate_preference_data(
            data=actions, limit=8
        )

        action_pref_data = ActionPairsPrefPairsContainer(
            action_pairs_data=action_data, pref_pairs_data=pref_data
        )

        memory = RoundsMemory(limit=3)

        memory.add_data(action_pref_data)

        data = memory.get_data_from_memory()

        assert len(memory.memory_list) == 1
        assert data.action_pairs_data.action_pairs.shape[0] == 8
        assert data.pref_pairs_data.preference_pairs.shape[0] == 8

        memory.add_data(action_pref_data)
        memory.add_data(action_pref_data)
        memory.add_data(action_pref_data)
        memory.add_data(action_pref_data)

        data = memory.get_data_from_memory()

        assert len(memory.memory_list) == 3
        assert data.action_pairs_data.action_pairs.shape[0] == 24
        assert data.pref_pairs_data.preference_pairs.shape[0] == 24
