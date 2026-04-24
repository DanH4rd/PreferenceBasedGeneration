from src.Abstract.AbsFeedbackSource import AbsFeedbackSource
from src.Abstract.AbsMemory import AbsMemory
from src.Abstract.AbsRewardModel import AbsRewardModel
from src.Abstract.AbsTrainer import AbsTrainer
from src.Abstract.AbsGenModel import AbsGenModel
from src.Abstract.AbsPreferenceDataGenerator import AbsPreferenceDataGenerator
from src.Abstract.AbsActionDistribution import AbsActionDistribution

from builder.AbsBuilder import AbsBuilder


class ConfigBuilder(AbsBuilder):
    """Builds pipeline components from Configuration dataclasses."""

    def create_reward_model(self, config) -> AbsRewardModel:
        if self.reward_model is not None:
            raise Exception("Reward model already set up")

        self.reward_model = self._create_component_from_config(config)
        self._is_object_of_required_class(AbsRewardModel, self.reward_model)

        return self.reward_model

    def create_gen_model(self, config) -> AbsGenModel:
        if self.gen_model is not None:
            raise Exception("Generator model already set up")

        self.gen_model = self._create_component_from_config(config)
        self._is_object_of_required_class(AbsGenModel, self.gen_model)

        return self.gen_model

    def create_memory(self, config) -> AbsMemory:
        if self.memory is not None:
            raise Exception("Memory already set up")

        self.memory = self._create_component_from_config(config)
        self._is_object_of_required_class(AbsMemory, self.memory)

        return self.memory

    def create_feedback_source(self, config) -> AbsFeedbackSource:
        if self.feedback_source is not None:
            raise Exception("Feedback source already set up")

        self.feedback_source = self._create_component_from_config(config)
        self._is_object_of_required_class(AbsFeedbackSource, self.feedback_source)

        return self.feedback_source

    def create_preference_generator(self, config) -> AbsPreferenceDataGenerator:
        if self.preference_generator is not None:
            raise Exception("Preference generator already set up")

        self.preference_generator = self._create_component_from_config(config)
        self._is_object_of_required_class(
            AbsPreferenceDataGenerator, self.preference_generator
        )

        return self.preference_generator

    def create_action_distribution(self, config) -> AbsActionDistribution:
        if self.action_distribution is not None:
            raise Exception("Action distribution already set up")

        self.action_distribution = self._create_component_from_config(config)
        self._is_object_of_required_class(
            AbsActionDistribution, self.action_distribution
        )

        return self.action_distribution

    def create_reward_model_trainer(self, config) -> AbsTrainer:
        if self.reward_model_trainer is not None:
            raise Exception("Reward model trainer already set up")

        self.reward_model_trainer = self._create_component_from_config(config)
        self._is_object_of_required_class(AbsTrainer, self.reward_model_trainer)

        return self.reward_model_trainer

    def create_destination_action_trainer(self, config) -> AbsTrainer:
        if self.destination_action_trainer is not None:
            raise Exception("Destination action trainer already set up")

        self.destination_action_trainer = self._create_component_from_config(config)
        self._is_object_of_required_class(AbsTrainer, self.destination_action_trainer)

        return self.destination_action_trainer
