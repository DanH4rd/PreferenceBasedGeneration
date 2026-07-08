import torch
from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal

from src.Abstract.AbsActionDistribution import AbsActionDistribution
from src.Abstract.AbsData import AbsData
from src.DataStructures import ActionData, TrainableActionData


class GreedyNormalActionDistribution(AbsActionDistribution):
    """Basically a torch.distribution.Distribution wrapper with extra functionality'

    Expects destination_action_trainable to be mutable and updated outside the class
    """

    def __init__(
        self,
        dist: Distribution,
        destination_action_trainable: TrainableActionData,
        e: float,
        decay_factor: float,
        omega2: float,
    ):
        self.dist = dist
        self.destination_action_trainable = destination_action_trainable
        self.e = e
        self.decay_factor = decay_factor
        self.omega2 = omega2

        self.nearby_dist = Normal(
            self.destination_action_trainable.detached_actions[0],
            torch.ones(self.destination_action_trainable.detached_actions[0].shape)
            * self.omega2,
            validate_args=None,
        )

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

        random_sample_size = int(N * self.e)

        for _ in range(random_sample_size):
            actions.append(self.dist.sample())

        for _ in range(N - random_sample_size):
            actions.append(self.dist.sample())

        return ActionData(actions=torch.stack(actions, dim=0))

    def update(self, data: AbsData | None) -> None:
        """Empty function since this class
        does not support updating based on
        provided data

        Args:
            data (AbsData | None): abstract data object
        """

        if len(self.destination_action_trainable.detached_actions) > 1:
            raise NotImplementedError(
                "GreedyNormalActionDistribution expects destination_action to be a single action"
            )

        self.e *= self.decay_factor

        self.nearby_dist = Normal(
            self.destination_action_trainable.detached_actions[0],
            torch.ones(self.destination_action_trainable.detached_actions[0].shape)
            * self.omega2,
            validate_args=None,
        )
        pass

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """
        return "Greedy Normal Action Distribution"
