import torch

from src.Abstract.AbsLoss import AbsLoss
from src.Abstract.AbsRewardModel import AbsRewardModel
from src.DataStructures import (
    ActionData,
    ActionPairsPrefPairsContainer,
)


class PreferenceLoss(AbsLoss[ActionPairsPrefPairsContainer]):
    """Class than calculates cross entrpy loss for preference probabilities
    for action pairs. Extimates preference probabilities based on action
    rewards
    """

    def __init__(self, rewardModel: AbsRewardModel, decimals: None | int = None):
        """
        Args:
            rewardModel (AbsRewardModel): reward model to get action rewards from
            decimals (None | int, optional): if not None will round preference values to the given
                decimals number. Defaults to None.

        Raises:
            Exception: if given decimals value is less than 1 or not an integer
        """

        self.decimals = decimals
        self.rewardModel = rewardModel

        if self.decimals is not None:
            if self.decimals < 1 or not isinstance(self.decimals, int):
                raise Exception(f"Invalid decimals value: {self.decimals}")

    def ConvertRewards2Preferences(
        self, r1: torch.Tensor, r2: torch.Tensor
    ) -> torch.Tensor:
        """Function that converts rewards pairs to preferences using SoftMax

        Args:
            r1 (torch.Tensor): list of first elements in reward pairs ([B,1] tensor),
                B - number of pairs
            r2 (torch.Tensor): list of second elements in reward pairs ([B,1] tensor)
                B - number of pairs

        Returns:
            torch.Tensor: preference probabilities for the first elements in pairs
        """

        # equivalent to exp(r1)/(exp(r1)+exp(r2)) but via sigmoid(r1-r2), which only
        # depends on the (finite) difference - the raw exp() form overflows to inf
        # once r1/r2 diverge by ~88 (float32), giving inf/inf = nan
        return torch.sigmoid(r1 - r2)

    def calculate_loss(self, data: ActionPairsPrefPairsContainer) -> torch.Tensor:
        """Calculates Cross Entropy loss on preference probabilities for the given
        action pairs and real preferences

        Args:
            data (ActionPairsPrefPairsContainer): object containing list of action
                pairs and corresponding list of real preference values, serving
                as ground truth labels

        Returns:
            torch.Tensor: mean of cross entropy loss with a grad
        """

        action_pairs_tensor = data.action_pairs_data.action_pairs
        pref_pairs_tensor = data.pref_pairs_data.preference_pairs
        sample_weights = data.sample_weights

        rewards_left_column = self.rewardModel.get_stable_rewards(
            ActionData(actions=action_pairs_tensor[:, 0, :])
        ).squeeze(1)
        rewards_right_column = self.rewardModel.get_stable_rewards(
            ActionData(actions=action_pairs_tensor[:, 1, :])
        ).squeeze(1)

        preferences_left_column = self.ConvertRewards2Preferences(
            rewards_left_column, rewards_right_column
        )

        if self.decimals is not None:
            preferences_left_column = torch.round(
                preferences_left_column, decimals=self.decimals
            )

        # clamp away from exact 0/1: a saturated probability times a 0-weight
        # preference label (e.g. skip pairs labelled [0,0]) computes 0 * log(0)
        # = 0 * -inf = nan below, poisoning the whole batch loss via .sum()
        preferences_left_column = preferences_left_column.clamp(1e-7, 1 - 1e-7)

        preferences_right_column = (
            torch.ones_like(preferences_left_column) - preferences_left_column
        )

        per_pair_loss = pref_pairs_tensor[..., 0] * torch.log(
            preferences_left_column
        ) + pref_pairs_tensor[..., 1] * torch.log(preferences_right_column)

        loss = -(per_pair_loss * sample_weights).sum() / sample_weights.sum()

        return loss

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """
        return f"Preference loss. Round to decimals: {self.decimals}"
