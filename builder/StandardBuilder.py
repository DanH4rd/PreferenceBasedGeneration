import yaml
from importlib import import_module

from src.Abstract.AbsFeedbackSource import AbsFeedbackSource
from src.Abstract.AbsMemory import AbsMemory
from src.Abstract.AbsRewardModel import AbsRewardModel
from src.Abstract.AbsTrainer import AbsTrainer
from src.Abstract.AbsActionFilter import AbsActionFilter
from src.Abstract.AbsGenModel import AbsGenModel
from src.Abstract.AbsPreferenceDataGenerator import AbsPreferenceDataGenerator
from src.Abstract.AbsLoss import AbsLoss
from src.Abstract.AbsActionDistribution import AbsActionDistribution

from src.Filter.CompositeSeriesActionFilter import CompositeActionFilter
from src.Loss.CompositeLoss import CompositeLoss



class StandardBuilder:
    """Class that handles the creation and basic dependencies of pipeline components
    """

    def __init__(self):
        self.gen_model = None
        self.reward_model = None
        self.action_distribution = None
        self.action_filters = CompositeActionFilter()

        self.preference_generator = None
        self.feedback_source = None
        self.memory = None

        self.reward_model_losses = CompositeLoss()
        self.reward_model_trainer = None

        self.destination_action_losses = CompositeLoss()
        self.destination_action_trainer = None

    def add_action_filter(self, action_filter: AbsActionFilter) -> AbsActionFilter:
        self.action_filters.add_filter(action_filter)
        return self.action_filters
    
    def add_reward_model_loss(self, loss: AbsLoss) -> AbsLoss:
        self.reward_model_losses.add_loss(loss)
        return self.reward_model_losses
    
    def add_destination_action_loss(self, loss: AbsLoss) -> AbsLoss:
        self.destination_action_losses.add_loss(loss)
        return self.destination_action_losses

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
        self._is_object_of_required_class(AbsPreferenceDataGenerator, self.preference_generator)

        return self.preference_generator
    
    def create_action_distribution(self, config) -> AbsActionDistribution:
        if self.action_distribution is not None:
            raise Exception("Action distribution already set up")
        
        self.action_distribution = self._create_component_from_config(config)
        self._is_object_of_required_class(AbsActionDistribution, self.action_distribution)

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
    
    def _create_component_from_config(self, config):
        """Dynamically loads the class of which config object
        is inner class from and calls its static function
        for creation of an instance of said class

        Args:
            config (dataclass): dataclass with constructor parametres for component

        Returns:
            dynamically created component
        """
        config_related_module = type(config).__module__
        config_related_class = str(type(config)).split('.')[-2]
        
        component = getattr(import_module(config_related_module), config_related_class).create_from_configuration(config)

        return component
    
    def _is_object_of_required_class(self, class_type:type, instance_obj:object):
        """Raises exception if instance object is not of or does not inferit the class_type

        Args:
            class_type (type): class type to compare to
            instance_obj (object): object to check

        Raises:
            Exception: if instance object is not of or does not inherit class type
        """
        if issubclass(class_type, type(instance_obj)) :
            raise Exception (f"Wrong component configuration object for {class_type}: {type(instance_obj)}")

