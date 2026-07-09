from dataclasses import dataclass

from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import make_grid

from src.Abstract.AbsActionDistribution import AbsActionDistribution
from src.Abstract.AbsActionFilter import AbsActionFilter
from src.Abstract.AbsPreferenceDataGenerator import AbsPreferenceDataGenerator
from src.Abstract.AbsRewardModel import AbsRewardModel
from src.Abstract.AbsTrainer import AbsTrainer
from src.DataStructures import (
    ActionData,
    ActionPairsPrefPairsContainer,
    TrainableActionData,
)
from src.FeedbackSource import CosDistFeedback, RandomFeedbackSource
from src.Filter import ScoreActionFilter
from src.GenModel import StackGanGenModel
from src.Memory import RoundsMemory
from src.MetricsLogger import TensorboardGridImageLogger, TensorboardImageLogger
from src.PreferenceDataGenerator import RandomPreferenceDataGenerator


class PbRLPipeline:
    """Encapsulates the PbRL training loop and post-loop control logging.

    Construction of components with live object-level dependencies (e.g.
    destination_action_trainable, which must be the same instance shared with
    GreedyNormalActionDistribution and ptLightningLatentWrapper) is the
    caller's responsibility and must be passed in already wired. Standalone
    helper components with no such cross-object dependencies (dummy data for
    the latent trainer contract, control filters/loggers for debug logging)
    are constructed internally by __init__ for convenience.

    Note on destination_action_trainable aliasing: the same TrainableActionData
    instance must be passed here, to GreedyNormalActionDistribution, and to
    ptLightningLatentWrapper. It registers its actions as an nn.Parameter, so
    the latent trainer optimises it in place directly (no copy-back step);
    the distribution reads its detached_actions[0] on update(). All three
    objects must share the same reference for mutations to propagate
    correctly across rounds.
    """

    DUMMY_SAMPLE_SIZE = 2
    PREFERENCE_DUMMY_LIMIT = 1

    @dataclass
    class Configuration:
        rounds: int
        samples_per_round: int
        preference_limit_per_round: int
        training_epochs_reward: int
        training_epochs_latent: int
        control_sample_size: int

    def __init__(
        self,
        config: "PbRLPipeline.Configuration",
        gen_model: StackGanGenModel,
        action_dist: AbsActionDistribution,
        destination_action_trainable: TrainableActionData,
        preference_generator: AbsPreferenceDataGenerator,
        memory: RoundsMemory,
        model_trainer: AbsTrainer,
        reward_model: AbsRewardModel,
        latent_trainer: AbsTrainer,
        sampling_filter: AbsActionFilter,
        tensorboard_writer: SummaryWriter,
    ):
        self.config = config
        self.gen_model = gen_model
        self.action_dist = action_dist
        self.destination_action_trainable = destination_action_trainable
        self.preference_generator = preference_generator
        self.memory = memory
        self.model_trainer = model_trainer
        self.latent_trainer = latent_trainer
        self.sampling_filter = sampling_filter

        # Log the first image and target image if
        tensorboard_writer.add_image(
            "Image/Starting Desc action",
            make_grid(
                gen_model.generate(
                    ActionData(actions=destination_action_trainable.detached_actions)
                ).images,
                nrow=1,
            ),
            0,
        )

        feedback_source = self.preference_generator.feedbackSource
        if isinstance(feedback_source, CosDistFeedback):
            tensorboard_writer.add_image(
                "Image/Target", pil_to_tensor(feedback_source.target_image), 0
            )
        # empty data needed for ptTrainer contract for optimising destination action
        dummy_preference_generator = RandomPreferenceDataGenerator(
            feedbackSource=RandomFeedbackSource()
        )
        self.dummy_action_data, self.dummy_pref_data = (
            dummy_preference_generator.generate_preference_data(
                data=self.gen_model.sample_random_actions(self.DUMMY_SAMPLE_SIZE),
                limit=self.PREFERENCE_DUMMY_LIMIT,
            )
        )

        # define filters for logging the best/worst actions according to the reward model each round for debug
        self.control_max_filter = ScoreActionFilter(
            mode="max", key=lambda x: reward_model.get_stable_rewards(x), limit=10
        )
        self.control_min_filter = ScoreActionFilter(
            mode="min", key=lambda x: reward_model.get_stable_rewards(x), limit=10
        )

        # configure round loggers
        self.round_image_logger = TensorboardImageLogger(
            name="Image/Destination Image", writer=tensorboard_writer
        )
        self.control_max_logger = TensorboardGridImageLogger(
            name="Image/Control Max", writer=tensorboard_writer, nrow=3
        )
        self.control_min_logger = TensorboardGridImageLogger(
            name="Image/Control Min", writer=tensorboard_writer, nrow=3
        )

    def run(self) -> None:
        for _ in range(self.config.rounds):
            self._run_round()
        self._run_control_logging()

    def _run_round(self) -> None:
        sampled = self.action_dist.sample(self.config.samples_per_round)
        sampled = self.sampling_filter.filter(action_data=sampled)
        sampled.append(self.destination_action_trainable.detached_actions)

        action_data, pref_data = self.preference_generator.generate_preference_data(
            data=sampled, limit=self.config.preference_limit_per_round
        )
        self.memory.add_data(
            ActionPairsPrefPairsContainer(
                action_pairs_data=action_data, pref_pairs_data=pref_data
            )
        )
        self.action_dist.update(None)

        train_data = self.memory.get_data_from_memory()

        self.model_trainer.run_training(
            action_data=train_data.action_pairs_data,
            preference_data=train_data.pref_pairs_data,
            epochs=self.config.training_epochs_reward,
            sample_weights=train_data.sample_weights,
        )

        self.latent_trainer.run_training(
            action_data=self.dummy_action_data,
            preference_data=self.dummy_pref_data,
            epochs=self.config.training_epochs_latent,
        )
        self.round_image_logger.log(
            self.gen_model.generate(
                ActionData(actions=self.destination_action_trainable.detached_actions)
            )
        )

    def _run_control_logging(self) -> None:
        for mode_filter, logger in [
            (self.control_max_filter, self.control_max_logger),
            (self.control_min_filter, self.control_min_logger),
        ]:
            control_actions = mode_filter.filter(
                self.gen_model.sample_random_actions(self.config.control_sample_size)
            )
            logger.log(self.gen_model.generate(control_actions))

    def __str__(self) -> str:
        return f"PbRLPipeline (rounds={self.config.rounds})"
