import abc
from dataclasses import dataclass

import torch

from GenerativeModelsData.StackGan2.StackGanUtils.config import cfg_from_file
from GenerativeModelsData.StackGan2.StackGanUtils.model import (
    D_NET64,
    D_NET128,
    D_NET256,
    D_NET1024,
)
from src.DataStructures import ImageData


class StackGanDiscModel(object, metaclass=abc.ABCMeta):
    """Adapter class for StackGanv2 implementation"""

    level_to_model = {
        0: D_NET64,
        1: D_NET128,
        2: D_NET256,
        3: D_NET1024,
    }

    @dataclass
    class Configuration:
        """dataclass for grouping constructor parametres"""

        config_file: str
        checkpoint_file: str
        scale_level: int
        ngpu: int = 1

    @staticmethod
    def create_from_configuration(conf: Configuration):
        return StackGanDiscModel(
            config_file=conf.config_file,
            checkpoint_file=conf.checkpoint_file,
            scale_level=conf.scale_level,
            ngpu=conf.ngpu,
        )

    def __init__(
        self, config_file: str, checkpoint_file: str, scale_level: int, ngpu: int = 1
    ):
        """

        Args:
            config_file (str): StackGanv2 path describing used model
                architecture parametres
            checkpoint_file (str): path to the checkpoint file to
                load model weights from
            scale_level (int): defines scale of image we want to work with
            ngpu (int, optional): number of gpus for torch.nn.DataParallel. Defaults to 1.
        """
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.scale_level = scale_level
        self.ngpu = ngpu
        self.device = "cpu"

        cfg_from_file(self.config_file)

        # create a model according to the provided scale level
        self.model = self.level_to_model[self.scale_level]()

        self.model = torch.nn.DataParallel(self.model, list(range(ngpu)))
        self.model.load_state_dict(
            torch.load(
                self.checkpoint_file, map_location=self.device, weights_only=True
            )
        )
        self.model.eval()

    def SetDevice(self, device) -> None:
        """Sets the device for discriminator model. The model is already
        wrapped in torch.nn.DataParallel at construction time (see ngpu), but
        DataParallel's own device_ids only make sense for CUDA devices -
        forward() requires every parameter to already sit on device_ids[0]
        when device_ids is non-empty. So device_ids is kept in sync here:
        cleared for a CPU device (DataParallel then calls the module
        directly, skipping its device checks) and restored to `ngpu`
        devices starting at the requested device's index otherwise.

        Args:
            device (_type_): device object to set to
        """

        self.device = device
        resolved_device = torch.device(device)

        # self.model is typed as nn.Module, whose __setattr__ stub only
        # accepts Tensor | Module - but at runtime it's a DataParallel
        # instance and device_ids/src_device_obj are its own plain attributes.
        if resolved_device.type == "cpu":
            self.model.device_ids = []  # pyright: ignore[reportArgumentType]
        else:
            primary_idx = (
                resolved_device.index if resolved_device.index is not None else 0
            )
            self.model.device_ids = list(  # pyright: ignore[reportArgumentType]
                range(primary_idx, primary_idx + self.ngpu)
            )
            self.model.src_device_obj = torch.device(  # pyright: ignore[reportArgumentType]
                resolved_device.type, primary_idx
            )

        self.model = self.model.to(device)

    def discriminate(self, data: ImageData) -> torch.Tensor:
        """Generates values based on provided image data

        Args:
            data (ImageData): images for which we want to
                calculate discriminator score

        Returns:
            torch.Tensor: [N,1] tensor containing discriminator scores
                for corresponding images
        """
        return self.model(data.images.to(self.device))[0]

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """
        return f"StackGan discriminator. \nConfig: {self.config_file}\nCheckpoint: {self.checkpoint_file}"
