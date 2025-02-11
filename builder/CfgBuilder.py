import yaml

from src.Abstract.AbsFeedbackSource import AbsFeedbackSource
from src.Abstract.AbsMemory import AbsMemory
from src.Abstract.AbsRewardModel import AbsRewardModel
from src.Abstract.AbsTrainer import AbsTrainer
from src.Abstract.AbsActionFilter import AbsActionFilter

from src.Filter.CompositeSeriesActionFilter import CompositeActionFilter
from src.Loss.CompositeLoss import CompositeLoss


class Hparam:
    default_param_dict = {
        "action_dimention_size": 100,
        "reward_model_type": "mlp_reward_model",
        "mlp_reward_model": {
            "p": 0.5,
            "input_dim": 100,
            "hidden_dim": 300,
        },
    }

    def __init__(self, cfg_file: str):

        if cfg_file != "":
            with open(cfg_file) as stream:
                self.config_file_params = yaml.safe_load(stream)


class CfgBuilder:
    """Class that builds the preference generation system
    from a config file
    """

    def __init__(self, cfg_file: str = ""):
        """
        Args:
            cfg_file (str): path of the the yaml config file
        """
        self.cfg_file = cfg_file

        self.hparam = Hparam(self.cfg_file)

        self.reward_model = None
        self.action_distribution = None
        self.action_filters = CompositeActionFilter()
        self.preference_generator = None
        self.memory = None
        self.reward_model_losses = CompositeLoss()
        self.reward_model_trainer = None
        self.destination_action_trainer = None


    def add_action_filter(self) -> AbsActionFilter:
        pass

    def create_reward_model(self) -> AbsRewardModel:
        pass

    def create_memory(self) -> AbsMemory:
        pass

    def create_feedback_generator(self) -> AbsFeedbackSource:
        pass
