import pytest

from src.FeedbackSource import RandomFeedbackSource
from src.PreferenceDataGenerator import (
    BestActionTracker,
    GraphPreferenceDataGeneration,
    RandomPreferenceDataGenerator,
)

feedback_source = RandomFeedbackSource()
pref_gen = []


pref_gen.append(RandomPreferenceDataGenerator(feedbackSource=feedback_source))
pref_gen.append(GraphPreferenceDataGeneration(feedbackSource=feedback_source))


class TestPreferenceDataGenerator:
    @pytest.mark.parametrize(
        "pref_gen,pref_gen_name", zip(pref_gen, map(str, pref_gen))
    )
    def test_basic(self, pref_gen, pref_gen_name, facade_gen_model):

        actions = facade_gen_model.sample_random_actions(5)

        action_pairs, preference_data = pref_gen.generate_preference_data(
            data=actions, limit=15
        )

        assert list(action_pairs.action_pairs.shape) == [
            10,
            2,
            actions.actions.shape[1],
        ]
        assert list(preference_data.preference_pairs.shape) == [10, 2]

        actions = facade_gen_model.sample_random_actions(45)

        action_pairs, preference_data = pref_gen.generate_preference_data(
            data=actions, limit=10
        )

        assert list(action_pairs.action_pairs.shape) == [
            10,
            2,
            actions.actions.shape[1],
        ]
        assert list(preference_data.preference_pairs.shape) == [10, 2]

    ###########

    @pytest.mark.xfail(
        reason="action loss while converting action pairs to action list with torch.unique",
        raises=AssertionError,
    )
    def test_best_action_tracker(self, facade_gen_model):

        feedback_source = RandomFeedbackSource()
        prefDataGenerator = RandomPreferenceDataGenerator(
            feedbackSource=feedback_source
        )
        prefDataGenerator = BestActionTracker(prefDataGen=prefDataGenerator)

        actions = facade_gen_model.sample_random_actions(15)

        action_pairs, preference_data = prefDataGenerator.generate_preference_data(
            data=actions, limit=10
        )

        # Sometimes BestActionTraker may lose some actions in the final stage of additional data
        # generation, check todo in BestActionTracker implementation
        assert list(action_pairs.action_pairs.shape) == [
            24,
            2,
            actions.actions.shape[1],
        ]  # 10 usual + 14 aditional pairs
        # (best + each other action)
        assert list(preference_data.preference_pairs.shape) == [24, 2]
