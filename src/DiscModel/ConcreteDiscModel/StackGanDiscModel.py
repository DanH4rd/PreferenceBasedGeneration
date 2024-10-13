import abc

import torch

from GenerativeModelsData.StackGan2.StackGanUtils.config import cfg_from_file
from GenerativeModelsData.StackGan2.StackGanUtils.model import (
    D_NET64,
    D_NET128,
    D_NET256,
    D_NET1024,
)
from src.DataStructures.ConcreteDataStructures.ImageData import ImageData


class StackGanDiscModel(object, metaclass=abc.ABCMeta):
    """
    Base class incupsulating the required logic for StackGanv2 model
    """

    def __init__(self, config_file, checkpoint_file, gen_level, ngpu=1):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.gen_level = gen_level
        self.device = "cpu"

        cfg_from_file(self.config_file)

        if self.gen_level == 0:
            self.model = D_NET64()
        elif self.gen_level == 1:
            self.model = D_NET128()
        elif self.gen_level == 2:
            self.model = D_NET256()
        elif self.gen_level == 3:
            self.model = D_NET256()
        elif self.gen_level == 4:
            self.model = D_NET1024()

        self.model = torch.nn.DataParallel(self.model, list(range(ngpu)))
        self.model.load_state_dict(
            torch.load(
                self.checkpoint_file, map_location=self.device, weights_only=True
            )
        )
        self.model.eval()

    def Discriminate(self, data: ImageData) -> torch.tensor:
        """
        Generates values based on provided action data
        """
        return self.model(data.images)[0]

    def __str__(self) -> str:
        return f"StackGan model. \nConfig: {self.config_file}\nCheckpoint: {self.checkpoint_file}"
