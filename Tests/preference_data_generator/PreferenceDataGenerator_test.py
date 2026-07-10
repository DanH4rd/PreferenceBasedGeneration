import pytest
import torch

from src.Abstract.AbsFeedbackSource import AbsFeedbackSource
from src.Abstract.AbsPreferenceDataGenerator import AbsPreferenceDataGenerator
from src.DataStructures import ActionData, ActionPairsData, PreferencePairsData
from src.FeedbackSource import RandomFeedbackSource
from src.PreferenceDataGenerator import (
    BestActionTracker,
    GraphPreferenceDataGeneration,
    RandomPreferenceDataGenerator,
)


class _AlwaysTieFeedbackSource(AbsFeedbackSource):
    """Deterministic feedback source: every pair ties, so best_action never changes"""

    def generate_feedback(
        self, action_pairs_data: ActionPairsData
    ) -> PreferencePairsData:
        pairs_num = action_pairs_data.action_pairs.shape[0]
        preference_tensor = torch.tensor([[0.5, 0.5]] * pairs_num)
        return PreferencePairsData(preference_pairs=preference_tensor)

    def __str__(self) -> str:
        return "Always tie feedback"


class _FixedPairPreferenceDataGenerator(AbsPreferenceDataGenerator):
    """Test double: always returns the same fixed pairs/preferences,
    regardless of `data`/`limit`, so best-action resolution is deterministic."""

    def __init__(self, feedbackSource, pairs_idx, preferences):
        self.feedbackSource = feedbackSource
        self._pairs_idx = pairs_idx
        self._preferences = preferences

    def generate_preference_data_idx(self, data, limit):
        return self._pairs_idx.clone(), PreferencePairsData(
            preference_pairs=self._preferences.clone()
        )

    def __str__(self) -> str:
        return "Fixed pair preference data generator"


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

        usual_pairs = 10
        max_additional_pairs = 14  # best vs each of the other 14 actions

        # additional pairs already directly compared with best in the usual
        # pairs are deduped, so the total can be less than the max (see
        # BestActionTracker.generate_preference_data_idx)
        total_pairs = action_pairs.action_pairs.shape[0]
        assert usual_pairs <= total_pairs <= usual_pairs + max_additional_pairs
        assert list(action_pairs.action_pairs.shape[1:]) == [
            2,
            actions.actions.shape[1],
        ]
        assert list(preference_data.preference_pairs.shape) == [total_pairs, 2]

    def test_best_action_tracker_dedupes_pairs_already_compared_with_best(self):
        """When the wrapped generator's own pairs already directly compare
        every other action against what becomes the best action, no redundant
        best-vs-action pairs should be added on top."""

        actions = ActionData(actions=torch.rand(5, 3))

        # action 0 beats every other action directly in the original data
        pairs_idx = torch.tensor([[0, 1], [0, 2], [0, 3], [0, 4]])
        preferences = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

        fixed_gen = _FixedPairPreferenceDataGenerator(
            feedbackSource=_AlwaysTieFeedbackSource(),
            pairs_idx=pairs_idx,
            preferences=preferences,
        )
        tracker = BestActionTracker(prefDataGen=fixed_gen)

        action_pairs_idx, preference_data = tracker.generate_preference_data_idx(
            data=actions, limit=10
        )

        # no additional (-1, idx) pairs appended: every other action was
        # already directly compared against best (idx 0) in the original data
        assert torch.equal(action_pairs_idx, pairs_idx)
        assert torch.equal(preference_data.preference_pairs, preferences)

    def test_best_action_tracker_resolves_carried_over_idx_across_rounds(
        self, facade_gen_model
    ):
        """best_action carried over from a previous round must be matched
        back to its idx in the new round's data, so it is excluded from
        self-comparison in the additional best-vs-rest pairs (see the
        `if best_action_idx is None:` block in BestActionTracker)."""

        feedback_source = _AlwaysTieFeedbackSource()
        prefDataGenerator = RandomPreferenceDataGenerator(
            feedbackSource=feedback_source
        )
        tracker = BestActionTracker(prefDataGen=prefDataGenerator)

        round_one_actions = facade_gen_model.sample_random_actions(15)
        tracker.generate_preference_data(data=round_one_actions, limit=10)

        tracked_best_action = tracker.best_action.clone()

        round_two_actions_tensor = facade_gen_model.sample_random_actions(10).actions
        match_idx = 3
        round_two_actions_tensor[match_idx] = tracked_best_action
        round_two_actions = ActionData(actions=round_two_actions_tensor)

        action_pairs_idx, _ = tracker.generate_preference_data_idx(
            data=round_two_actions, limit=10
        )

        # the row identical to the carried-over best action resolves to
        # best_action_idx, so it must never appear as the non-best side of
        # a best-vs-rest (-1, idx) self-comparison pair
        assert not (
            (action_pairs_idx == torch.tensor([-1, match_idx])).all(dim=1).any()
        )
