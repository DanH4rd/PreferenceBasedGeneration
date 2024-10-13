from src.Loss.AbsLoss import AbsLoss


class RewardValuesRegularisation(AbsLoss):

    def __str__(self) -> str:
        """
        Returns string describing the object
        """
        return "Reward Values Regularisator"
