from src.Abstract.AbsLoss import AbsLoss


class RewardValuesRegularisation(AbsLoss):
    """Regularization loss for reward model that punishes
    if the generated reward value is not in the given range
    """

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str: _description_
        """

        return "Reward Values Regularisator"
