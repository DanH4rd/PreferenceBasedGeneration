from Filter.AbsActionFilter import AbsActionFilter
import torch
from src.DataStructures.ConcreteDataStructures.ActionData import ActionData

class CompositeActionFilter(AbsActionFilter):
    """
        Filter that is a serial composition of several other filters, performing one by one
    """

    def __init__(self, filters:list[AbsActionFilter]):
        """
            Params:
                loggers - list of loggers out of which the composite consists of 
        """

        self.filters = filters
        self.limit = None


    def AddFilter(self, filter:AbsActionFilter|list[AbsActionFilter]) -> None:
        """
            Adds a filter to the composite elements list. Can accept a list
            of filters as a parametre, in this case it will concat
            the registered filters list with the passed filter lidt

            Params:
                filters - AbsLogger object or a list of those
        """

        if isinstance(filter, list):
            self.filter += filter
        else:
            self.filter.append(filter)

    def Filter(self, data:ActionData) -> ActionData:
        """
            Performs the Filter function of all composite elements.
            Filtering is performed one by one - the output of the 1st
            filter is the input of the 2nd filter

            Check the abstract base class for more info.
        """
        actions = data.actions
        for filter in self.filters:
            actions = filter.Filter(actions)

        return ActionData(actions=actions)

    def __str__(self) -> str:
        return f"Composite action filter. Number of members: {len(self.filter)}"