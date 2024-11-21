import torch

from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.DataStructures.ConcreteDataStructures.ActionPairsData import ActionPairsData
from src.DataStructures.ConcreteDataStructures.PreferencePairsData import (
    PreferencePairsData,
)
from src.PreferenceDataGenerator.AbsPreferenceDataGenerator import (
    AbsPreferenceDataGenerator,
)


class BestActionTracker(AbsPreferenceDataGenerator):
    """Decorator class that adds to the generated preferences
    of another preference generator new preferences created
    by tracking what action is the best of all met thus far 
    """

    def __init__(self, prefDataGen: AbsPreferenceDataGenerator):
        """
        Args:
            prefDataGen (AbsPreferenceDataGenerator): preference generator
                for which to add best action tracking functionality
        """        

        self.prefDataGen = prefDataGen

        self.best_action = torch.tensor([])

    def generate_preference_data(
        self, data: ActionData, limit: int
    ) -> tuple[ActionPairsData, PreferencePairsData]:
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
        
        
        TODO:
            add an option to ensure in generating additional data stage
            that generated preferences and action pairs are not already present in originally
            generated data

            (ACT-LOSS) (ctr f to find line) the operation to get a list of used actions sometimes
            loses some actions while converting from action pairs
        """        

        action_pairs_data, preference_data = self.prefDataGen.generate_preference_data(
            data=data, limit=limit
        )

        pref_tensor = preference_data.preference_pairs
        action_tensor = action_pairs_data.action_pairs

        best_action_impossible_candidates = []

        # get actions that are not able to be the best (unprefered actions)
        for i in range(pref_tensor.shape[0]):
            preference = pref_tensor[i]
            action_pair = action_tensor[i]

            if (preference == torch.tensor([0.0, 0.0])).all():
                best_action_impossible_candidates.append(action_pair[0])
                best_action_impossible_candidates.append(action_pair[1])
            elif (preference == torch.tensor([0.5, 0.5])).all():
                pass
            else:
                unpreferable_action_position = torch.argmin(preference)
                unpreferable_action = action_pair[unpreferable_action_position]
                best_action_impossible_candidates.append(unpreferable_action)

        best_action_impossible_candidates = torch.stack(
            best_action_impossible_candidates, dim=0
        ).unique(dim=0)

        # filter actions from impossible candidates
        candidate_actions = torch.flatten(action_tensor, start_dim=0, end_dim=1).unique(
            dim=0
        )

        for impossible_candidate in best_action_impossible_candidates:

            possible_candidate_pos_table = ~(
                (candidate_actions - impossible_candidate) < 1e-10
            ).all(dim=1)

            # if we have no candidates, return original data
            if ~possible_candidate_pos_table.any():
                return action_pairs_data, preference_data

            candidate_actions = candidate_actions[possible_candidate_pos_table]

        # if there is no best action (cold start), pick the first candidate as the best
        if self.best_action.shape[0] == 0:
            self.best_action = candidate_actions[0]
            candidate_actions = candidate_actions[1:]

        # based on more feedback, find the best action based on all of the candidates
        for candidate in candidate_actions:
            action_pair_tensor = torch.stack(
                [self.best_action, candidate], dim=0
            ).unsqueeze(0)
            action_pair_data = ActionPairsData(action_pairs=action_pair_tensor)

            pref_pair_data = self.prefDataGen.feedbackSource.generate_feedback(
                action_pair_data
            )
            preference = pref_pair_data.preference_pairs[0]
            # print(preference)

            if (preference == torch.tensor([0.0, 1.0])).all():
                # print('change')
                self.best_action = candidate

        # print(self.best_action)

        # generate new preference data

        all_actions = torch.flatten(action_tensor, start_dim=0, end_dim=1).unique(
            dim=0
        )  # (ACT-LOSS) sometimes loses some actions

        all_actions_without_best_idx_table = ~(
            (all_actions - self.best_action) < 1e-10
        ).all(dim=1)
        all_actions = all_actions[all_actions_without_best_idx_table]

        additional_action_pairs = []
        additional_pref_pairs = []

        for action in all_actions:
            action_pair = torch.stack([self.best_action, action], dim=0)
            additional_action_pairs.append(action_pair)
            additional_pref_pairs.append(torch.tensor([1.0, 0.0]))

        additional_action_pairs = torch.stack(additional_action_pairs)
        additional_pref_pairs = torch.stack(additional_pref_pairs)
        action_pairs_tensor = torch.concat(
            [action_pairs_data.action_pairs, additional_action_pairs], dim=0
        )
        action_prefs_tensor = torch.concat(
            [preference_data.preference_pairs, additional_pref_pairs], dim=0
        )

        action_pairs_data = ActionPairsData(action_pairs=action_pairs_tensor)
        preference_data = PreferencePairsData(preference_pairs=action_prefs_tensor)

        return action_pairs_data, preference_data

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """        
        return f"Best Action Tracker for {str(self.prefDataGen)}"
