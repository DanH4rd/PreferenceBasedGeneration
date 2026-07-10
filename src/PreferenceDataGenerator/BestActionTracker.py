from dataclasses import dataclass

import torch

from src.Abstract.AbsPreferenceDataGenerator import AbsPreferenceDataGenerator
from src.DataStructures import ActionData, ActionPairsData, PreferencePairsData


class BestActionTracker(AbsPreferenceDataGenerator):
    """Decorator class that adds to the generated preferences
    of another preference generator new preferences created
    by tracking what action is the best of all met thus far
    """

    @dataclass
    class Configuration:
        """dataclass for grouping constructor parametres"""

        prefDataGen: AbsPreferenceDataGenerator

    @staticmethod
    def create_from_configuration(conf: Configuration):
        return BestActionTracker(prefDataGen=conf.prefDataGen)

    def __init__(self, prefDataGen: AbsPreferenceDataGenerator):
        """
        Args:
            prefDataGen (AbsPreferenceDataGenerator): preference generator
                for which to add best action tracking functionality
        """

        self.prefDataGen = prefDataGen
        self.feedbackSource = prefDataGen.feedbackSource

        self.best_action = torch.tensor([])

    def generate_preference_data(
        self, data: ActionData, limit: int
    ) -> tuple[ActionPairsData, PreferencePairsData]:
        """Overload of the parent method that supports best_action logic"""
        pair_idx_tensor, preference_data = self.generate_preference_data_idx(
            data, limit
        )

        # best_action must be appended (not prepended) so idx -1 resolves to it
        # via negative indexing while 0-based idx from data.actions stay unshifted
        actions_tensor = torch.concat(
            [data.actions, self.best_action.unsqueeze(dim=0)], dim=0
        )
        action_pair_tensor = torch.stack(
            [
                actions_tensor[pair_idx_tensor[:, 0]],
                actions_tensor[pair_idx_tensor[:, 1]],
            ],
            dim=1,
        )
        action_pairs_data = ActionPairsData(action_pairs=action_pair_tensor)

        return action_pairs_data, preference_data

    def generate_preference_data_idx(
        self, data: ActionData, limit: int
    ) -> tuple[torch.Tensor, PreferencePairsData]:
        """Generates preference data with given preference data generator and
        asks for additional preference data to determine the best action.
        Generates additional preference data based on best actions.

        Args:
            data (ActionData): list of actions to generate preferences for

            limit (int): maximum number of preferences the generator can
                ask the feedback source for preferences. Does not apply to
                BestActionTracker number of requests to feedbackSource.

        Returns:
            tuple[ActionPairsData, PreferencePairsData]: list of action pairs with corresponding preferences
        """
        all_actions_tensor = data.actions
        action_pairs_idx, preference_data = (
            self.prefDataGen.generate_preference_data_idx(data=data, limit=limit)
        )

        pref_tensor = preference_data.preference_pairs

        best_action_candidates_idx = []

        # collect preferred (winning) actions from each decisive pair as best-action candidates
        for i in range(pref_tensor.shape[0]):
            preference = pref_tensor[i]
            action_pair_idx = action_pairs_idx[i]

            if (preference == torch.tensor([0.0, 0.0])).all():
                pass
            elif (preference == torch.tensor([0.5, 0.5])).all():
                # tied pair: neither action was ruled out, both remain candidates
                best_action_candidates_idx.append(action_pair_idx[0])
                best_action_candidates_idx.append(action_pair_idx[1])
            else:
                preferable_action_position = torch.argmax(preference)
                preferable_action = action_pair_idx[preferable_action_position]
                best_action_candidates_idx.append(preferable_action)

        # actions never asked about this round were never ruled out either;
        # treat them as untested candidates instead of silently excluding them
        known_action_idx = action_pairs_idx.flatten().unique()
        all_action_idx = torch.arange(all_actions_tensor.shape[0])
        unseen_action_idx = all_action_idx[
            ~torch.isin(all_action_idx, known_action_idx)
        ]
        best_action_candidates_idx.extend(unseen_action_idx)

        if best_action_candidates_idx:
            candidate_actions_idx = torch.stack(
                best_action_candidates_idx, dim=0
            ).unique(dim=0)
        else:
            candidate_actions_idx = torch.tensor([], dtype=torch.long)

        # if there is no set best action (cold start), pick the first candidate as the best
        best_action_idx = None
        if self.best_action.shape[0] == 0:
            if candidate_actions_idx.shape[0] > 0:
                best_action_idx = candidate_actions_idx[0]
                candidate_actions_idx = candidate_actions_idx[1:]
            else:
                # no preference signal at all yet, fall back to an arbitrary action
                best_action_idx = 0
            self.best_action = all_actions_tensor[best_action_idx]

        # best_action carried over unchanged from a previous round: resolve its idx
        # in this round's data (if present) so the self-comparison guard below works
        if best_action_idx is None:
            match_table = torch.isclose(all_actions_tensor, self.best_action).all(dim=1)
            match_idx = torch.nonzero(match_table, as_tuple=False)
            if match_idx.shape[0] > 0:
                best_action_idx = match_idx[0].item()

        # based on more feedback, find the best action based on all of the candidates
        for candidate_idx in candidate_actions_idx:
            action_pair_tensor = torch.stack(
                [self.best_action, all_actions_tensor[candidate_idx]], dim=0
            ).unsqueeze(0)
            action_pair_data = ActionPairsData(action_pairs=action_pair_tensor)

            pref_pair_data = self.prefDataGen.feedbackSource.generate_feedback(
                action_pair_data
            )
            preference = pref_pair_data.preference_pairs[0]
            # print(preference)

            if (preference == torch.tensor([0.0, 1.0])).all():
                # print('change')
                best_action_idx = candidate_idx
                self.best_action = all_actions_tensor[best_action_idx]

        # print(self.best_action)

        # generate new preference data, skipping any best-vs-action pair whose
        # actions were already directly compared in this round's originally
        # generated pairs (no need to ask the same comparison twice)
        best_action_idx_int = (
            int(best_action_idx) if best_action_idx is not None else None
        )
        existing_pairs = set()
        if best_action_idx_int is not None:
            for pair in action_pairs_idx:
                a_idx, b_idx = int(pair[0]), int(pair[1])
                existing_pairs.add((a_idx, b_idx))
                existing_pairs.add((b_idx, a_idx))

        new_pairs_idx_list = []
        new_prefs_list = []

        for action_idx, _ in enumerate(all_actions_tensor):
            # dont compare the best action with itself
            if best_action_idx_int is not None and action_idx == best_action_idx_int:
                continue
            # already compared against the best action in this round's original data
            if (
                best_action_idx_int is not None
                and (best_action_idx_int, action_idx) in existing_pairs
            ):
                continue
            new_pairs_idx_list.append(torch.tensor([-1, action_idx]))
            new_prefs_list.append(torch.tensor([1.0, 0.0]))

        if new_pairs_idx_list:
            additional_action_pairs_idx = torch.stack(new_pairs_idx_list)
            additional_pref_pairs = torch.stack(new_prefs_list)
        else:
            additional_action_pairs_idx = torch.empty((0, 2), dtype=torch.long)
            additional_pref_pairs = torch.empty(
                (0, 2), dtype=preference_data.preference_pairs.dtype
            )

        action_pairs_idx = torch.concat(
            [action_pairs_idx, additional_action_pairs_idx], dim=0
        )
        action_prefs_tensor = torch.concat(
            [preference_data.preference_pairs, additional_pref_pairs], dim=0
        )

        preference_data = PreferencePairsData(preference_pairs=action_prefs_tensor)

        return action_pairs_idx, preference_data

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """
        return f"Best Action Tracker for {str(self.prefDataGen)}"
