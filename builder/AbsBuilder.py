from abc import ABC
from importlib import import_module

from src.Abstract.AbsActionFilter import AbsActionFilter
from src.Abstract.AbsLoss import AbsLoss

from src.Filter.CompositeSeriesActionFilter import CompositeActionFilter
from src.Loss.CompositeLoss import CompositeLoss


class AbsBuilder(ABC):
    """Abstract base for pipeline builders. Holds shared component state and utilities."""

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

    def _create_component_from_config(self, config):
        """Dynamically loads the class whose config is a nested dataclass and instantiates it."""
        config_related_module = type(config).__module__
        config_related_class = str(type(config)).split(".")[-2]

        component = getattr(
            import_module(config_related_module), config_related_class
        ).create_from_configuration(config)

        return component

    def _is_object_of_required_class(self, class_type: type, instance_obj: object):
        if not isinstance(instance_obj, class_type):
            raise Exception(
                f"Wrong component configuration object for {class_type}: {type(instance_obj)}"
            )
