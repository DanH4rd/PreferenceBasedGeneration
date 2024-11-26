import yaml

from src.FeedbackSource.AbsFeedbackSource import AbsFeedbackSource
from src.Memory.AbsMemory import AbsMemory
from src.RewardModel.AbsRewardModel import AbsRewardModel
from src.Trainer.AbsTrainer import AbsTrainer


class Hparam:
    param_dict = {
        "action_dimention_size": 100,
        "reward_model_type": "mlp_reward_model",
        "mlp_reward_model": {
            "drop_out_chance": 0.5,
            "hidden_layer_divention_size": 100,
        },
    }

    def __init__(self, cfg_file: str):

        pass


class CfgBuilder:
    """Class that builds the preference generation system
    from a config file
    """

    def __init__(self, cfg_file: str):
        """
        Args:
            cfg_file (str): path the the yaml config file
        """
        self.cfg_file = cfg_file

        with open(self.cfg_file) as stream:
            self.hparam = yaml.safe_load(stream)

    def create_reward_model(self) -> AbsRewardModel:
        pass

    def create_trainer(self) -> AbsTrainer:
        pass

    def create_memory(self) -> AbsMemory:
        pass

    def create_feedback_generator(self) -> AbsFeedbackSource:
        pass
