from dataclasses import dataclass

from src.Abstract.AbsActionDistribution import AbsActionDistribution
from src.Abstract.AbsActionFilter import AbsActionFilter
from src.Abstract.AbsGenModel import AbsGenModel
from src.Abstract.AbsMemory import AbsMemory
from src.Abstract.AbsMetricsLogger import AbsMetricsLogger
from src.Abstract.AbsPreferenceDataGenerator import AbsPreferenceDataGenerator
from src.Abstract.AbsTrainer import AbsTrainer
from src.DataStructures.ActionData import ActionData
from src.DataStructures.ActionPairsPrefPairsContainer import ActionPairsPrefPairsContainer


class PbRLPipeline:
    """Encapsulates the PbRL training loop and post-loop control logging.

    Component construction and wiring — including objects with live object-level
    dependencies such as destination_action — is the caller's responsibility.
    This class only executes the pipeline given fully wired components.

    Note on destination_action aliasing: the same ActionData instance must be
    passed here and to both GreedyNormalActionDistribution and
    ptLightningLatentWrapper. The latent trainer writes optimized tensors back
    to destination_action.actions each epoch; the distribution reads
    destination_action.actions[0] on update(). All three objects must share the
    same reference for mutations to propagate correctly across rounds.
    """

    @dataclass
    class Configuration:
        rounds: int
        samples_per_round: int
        preference_limit_per_round: int
        training_epochs_reward: int
        training_epochs_latent: int
        dummy_sample_size: int
        preference_dummy_limit: int
        control_sample_size: int

    def __init__(
        self,
        config: "PbRLPipeline.Configuration",
        gen_model: AbsGenModel,
        action_dist: AbsActionDistribution,
        destination_action: ActionData,
        preference_generator: AbsPreferenceDataGenerator,
        dummy_preference_generator: AbsPreferenceDataGenerator,
        memory: AbsMemory,
        model_trainer: AbsTrainer,
        latent_trainer: AbsTrainer,
        sampling_filter: AbsActionFilter,
        control_max_filter: AbsActionFilter,
        control_min_filter: AbsActionFilter,
        round_image_logger: AbsMetricsLogger,
        control_max_logger: AbsMetricsLogger,
        control_min_logger: AbsMetricsLogger,
    ):
        self.config = config
        self.gen_model = gen_model
        self.action_dist = action_dist
        self.destination_action = destination_action
        self.preference_generator = preference_generator
        self.dummy_preference_generator = dummy_preference_generator
        self.memory = memory
        self.model_trainer = model_trainer
        self.latent_trainer = latent_trainer
        self.sampling_filter = sampling_filter
        self.control_max_filter = control_max_filter
        self.control_min_filter = control_min_filter
        self.round_image_logger = round_image_logger
        self.control_max_logger = control_max_logger
        self.control_min_logger = control_min_logger

    def run(self) -> None:
        for _ in range(self.config.rounds):
            self._run_round()
        self._run_control_logging()

    def _run_round(self) -> None:
        sampled = self.action_dist.sample(self.config.samples_per_round)
        sampled = self.sampling_filter.filter(action_data=sampled)
        sampled.append(self.destination_action.actions.detach())

        action_data, pref_data = self.preference_generator.generate_preference_data(
            data=sampled, limit=self.config.preference_limit_per_round
        )
        self.memory.add_data(
            ActionPairsPrefPairsContainer(
                action_pairs_data=action_data, pref_pairs_data=pref_data
            )
        )
        train_data = self.memory.get_data_from_memory()

        self.model_trainer.run_training(
            action_data=train_data.action_pairs_data,
            preference_data=train_data.pref_pairs_data,
            epochs=self.config.training_epochs_reward,
        )

        dummy_action_data, dummy_pref_data = (
            self.dummy_preference_generator.generate_preference_data(
                data=self.gen_model.sample_random_actions(self.config.dummy_sample_size),
                limit=self.config.preference_dummy_limit,
            )
        )
        self.action_dist.update(None)
        self.latent_trainer.run_training(
            action_data=dummy_action_data,
            preference_data=dummy_pref_data,
            epochs=self.config.training_epochs_latent,
        )
        self.round_image_logger.log(self.gen_model.generate(self.destination_action))

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
