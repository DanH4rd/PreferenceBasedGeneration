import yaml

from src.Abstract.AbsFeedbackSource import AbsFeedbackSource
from src.Abstract.AbsMemory import AbsMemory
from src.Abstract.AbsRewardModel import AbsRewardModel

from builder.AbsBuilder import AbsBuilder


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


class CfgBuilder(AbsBuilder):
    """Builds pipeline components from a YAML config file."""

    def __init__(self, cfg_file: str = ""):
        super().__init__()
        self.cfg_file = cfg_file
        self.hparam = Hparam(self.cfg_file)

    def create_reward_model(self) -> AbsRewardModel:
        pass

    def create_memory(self) -> AbsMemory:
        pass

    def create_feedback_source(self) -> AbsFeedbackSource:
        pass
