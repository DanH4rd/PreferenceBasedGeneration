import torch
from torch.distributions.distribution import Distribution

from src.ActionDistribution.AbsActionDistribution import AbsActionDistribution
from src.DataStructures.AbsData import AbsData
from src.DataStructures.ConcreteDataStructures.ActionData import ActionData


class SimpleActionDistribution(AbsActionDistribution):
    """Basically a torch.distribution.Distribution wrapper
    Doesnt do anything extra
    """

    def __init__(self, dist: Distribution):
        self.dist = dist

    def sample(self, N: int) -> ActionData:
        """Sample N actions from the distribution

        Args:
            N (int): number of actions to sample

        Raises:
            Exception: if the number of actions to
            sample is lower than zero

        Returns:
            ActionData: _description_
        """

        if N < 1:
            raise Exception(f"Sample number cannot be lower 1. Provided: {N}")

        actions = []

        for _ in range(N):
            actions.append(self.dist.sample())

        return ActionData(actions=torch.stack(actions, dim=0))

    def update(self, data: AbsData) -> None:
        """Empty function since this class
        does not support updating based on
        provided data

        Args:
            data (AbsData): abstract data object
        """
        pass

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """
        return "Simple Action Distribution"
