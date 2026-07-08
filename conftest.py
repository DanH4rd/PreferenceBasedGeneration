import os

import pytest

from src.DiscModel import StackGanDiscModel
from src.GenModel import StackGanGenModel

STACKGAN_CONFIG = "GenerativeModelsData/StackGan2/config/facade_3stages_color.yml"
STACKGAN_CHECKPOINT_G = (
    "GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netG_56500.pth"
)
STACKGAN_CHECKPOINT_D2 = (
    "GenerativeModelsData/StackGan2/checkpoints/Facade v1.0/netD2.pth"
)

FEEDBACK_TARGET_IMAGE = "Tests/feedback/images/ArtNouveaufacade79.jpeg"
METRICS_LOGGER_IMAGES_DIR = "Tests/metrics_logger/images"


def require_path(path: str) -> str:
    """Skips the test if a required external asset is missing.

    StackGan2 checkpoints/configs and Tests image fixtures are not
    checked into the repository (see CLAUDE.md / .gitignore), so a
    fresh checkout would otherwise hard-fail these tests.
    """
    if not os.path.exists(path):
        pytest.skip(f"Required test asset not found: {path}")
    return path


# StackGan2's `cfg` (GenerativeModelsData/StackGan2/StackGanUtils/config.py) is a
# module-level mutable singleton merged in-place by cfg_from_file() on every model
# construction and every get_input_noise_distribution() call. These fixtures are
# session-scoped and all load STACKGAN_CONFIG, so this is safe today — but a future
# fixture/test loading a different config file would silently mutate `cfg` for the
# rest of the session, affecting any test relying on it (e.g. Z_DIM) afterwards.
@pytest.fixture(scope="session")
def facade_gen_model() -> StackGanGenModel:
    return StackGanGenModel(
        config_file=require_path(STACKGAN_CONFIG),
        checkpoint_file=require_path(STACKGAN_CHECKPOINT_G),
        scale_level=2,
    )


@pytest.fixture(scope="session")
def facade_gen_model_scale0() -> StackGanGenModel:
    return StackGanGenModel(
        config_file=require_path(STACKGAN_CONFIG),
        checkpoint_file=require_path(STACKGAN_CHECKPOINT_G),
        scale_level=0,
    )


@pytest.fixture(scope="session")
def facade_disc_model() -> StackGanDiscModel:
    return StackGanDiscModel(
        config_file=require_path(STACKGAN_CONFIG),
        checkpoint_file=require_path(STACKGAN_CHECKPOINT_D2),
        scale_level=2,
    )


@pytest.fixture
def feedback_target_image_path() -> str:
    return require_path(FEEDBACK_TARGET_IMAGE)


@pytest.fixture
def metrics_logger_images_dir() -> str:
    return require_path(METRICS_LOGGER_IMAGES_DIR)
