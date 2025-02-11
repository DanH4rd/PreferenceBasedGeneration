import abc
import warnings
from dataclasses import dataclass

import torch
from torch.distributions.normal import Normal
from torchvision.utils import make_grid

from GenerativeModelsData.StackGan2.StackGanUtils.config import cfg, cfg_from_file
from GenerativeModelsData.StackGan2.StackGanUtils.model import G_NET
from src.Abstract.AbsMetricsLogger import AbsMetricsLogger
from src.DataStructures.ActionData import ActionData
from src.DataStructures.ImageData import ImageData


class StackGanGenModel(object, metaclass=abc.ABCMeta):
    """Adapter class for StackGanv2 generator"""

    @dataclass
    class Configuration:
        """dataclass for grouping constructor parametres
        """
        config_file: str
        checkpoint_file: str
        scale_level: int
        ngpu: int = 1

    @staticmethod
    def create_from_configuration(conf: Configuration):
        return StackGanGenModel(config_file= conf.config_file, 
                                 checkpoint_file=conf.checkpoint_file,
                                 scale_level=conf.scale_level,
                                 ngpu=conf.ngpu)

    def __init__(self, config_file, checkpoint_file, scale_level, ngpu=1):
        """
        Args:
            config_file (_type_): StackGanv2 path describing used model
                architecture parametres
            checkpoint_file (_type_): path to the checkpoint file to
                load model weights from
            scale_level (_type_): defines scale of image we want to work with
            ngpu (int, optional): number of gpus for torch.nn.DataParallel. Defaults to 1.
        """
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.scale_level = scale_level
        self.device = "cpu"

        cfg_from_file(self.config_file)
        self.model = G_NET()

        self.model = torch.nn.DataParallel(self.model, list(range(ngpu)))
        self.model.load_state_dict(
            torch.load(
                self.checkpoint_file, map_location=self.device, weights_only=True
            )
        )
        self.model.eval()

    def SetDevice(self, device) -> None:
        """Sets the device for generator model.

        Args:
            device (_type_): device object to set to

        TODO:
            support parralel
        """

        warnings.warn(
            "No other devices are supported for StackGan model", RuntimeWarning
        )

    def generate(self, data: ActionData) -> ImageData:
        """Generates images of set scale based on
        provided action action list

        Args:
            data (ActionData): List of actions to generate images for

        Returns:
            ImageData: generated images of said scale
        """
        images = self.model(data.actions)[0][self.scale_level]

        # normalise values of generated images
        images = torch.stack(
            list(map(lambda x: make_grid(x, padding=0, normalize=True), images))
        )

        return ImageData(images=images)

    def sample_random_actions(self, N: int) -> ActionData:
        """Samples given number of noise vectors (actions) from noise distribution

        Args:
            N (int): noise vectors number to generate

        Raises:
            Exception: If given number of actions to generate is lower than 1

        Returns:
            ActionData: list of sampled actions
        """

        if N < 1:
            raise Exception(f"Generate noise number cannot be lower 1. Provided: {N}")

        # cfg_from_file(self.config_file)

        # action_dim = cfg.GAN.Z_DIM

        # return ActionData(actions=torch.normal(
        #                             torch.zeros((N, action_dim)),
        #                             torch.ones((N, action_dim))
        #                                       )
        #                  )

        dist = self.get_input_noise_distribution()
        actions = []
        for _ in range(N):
            actions.append(dist.sample())

        return ActionData(actions=torch.stack(actions, dim=0))

    def get_input_noise_distribution(self) -> Normal:
        """Returns the torch distribution object, corresponding to the
        distribution of noise distribution used for input

        Returns:
            Normal: Normal distribution object
        """

        cfg_from_file(self.config_file)

        action_dim = cfg.GAN.Z_DIM

        dist = Normal(
            torch.tensor([0.0] * action_dim),
            torch.tensor([1.0] * action_dim),
            validate_args=None,
        )

        return dist

    def get_media_logger(self) -> AbsMetricsLogger:
        """Returns image logger to tensorboard

        Raises:
            NotImplementedError: Not implemented

        Returns:
            AbsMetricsLogger: Image tensorboard logger
        """

        raise NotImplementedError()

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str:
        """
        return f"StackGan model. \nConfig: {self.config_file}\nCheckpoint: {self.checkpoint_file}"
