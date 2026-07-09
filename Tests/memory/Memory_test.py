import torch

from src.DataStructures import (
    ActionPairsPrefPairsContainer,
)
from src.FeedbackSource import RandomFeedbackSource
from src.Memory import RoundsMemory
from src.PreferenceDataGenerator import (
    RandomPreferenceDataGenerator,
)


class TestMemory:
    def test_rounds_memory(self, facade_gen_model):
        gen_model = facade_gen_model
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

    def test_rounds_memory_discount_weights(self, facade_gen_model):
        """The discount factor must land as a per-pair sample weight, not be
        multiplied into the preference values (which would break
        PreferencePairsData's legal-value invariant for any pair but [0., 0.]).
        """
        gen_model = facade_gen_model
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

        memory = RoundsMemory(limit=3, discount_factor=0.9)

        memory.add_data(action_pref_data)
        memory.add_data(action_pref_data)
        memory.add_data(action_pref_data)

        data = memory.get_data_from_memory()

        # preference values themselves are untouched by discounting
        assert torch.equal(
            data.pref_pairs_data.preference_pairs,
            torch.concat([pref_data.preference_pairs] * 3, dim=0),
        )

        expected_weights = torch.concat(
            [
                torch.full((8,), 0.9**3),
                torch.full((8,), 0.9**2),
                torch.full((8,), 0.9**1),
            ]
        )
        assert torch.allclose(data.sample_weights, expected_weights)
