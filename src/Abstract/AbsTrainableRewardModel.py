import abc

from src.Abstract.AbsRewardModel import AbsRewardModel
from src.Abstract.AbsTrainableModel import AbsTrainableModel


class AbsTrainableRewardModel(AbsRewardModel, AbsTrainableModel, metaclass=abc.ABCMeta):
    """Reward model that also supports training-time concerns
    (mode switching, freezing, device placement).

    Used where a single model must be both queried for rewards and
    frozen/unfrozen during latent action optimisation.
    """
