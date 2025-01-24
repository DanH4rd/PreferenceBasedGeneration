from src.Abstract.AbsActionFilter import AbsActionFilter
from src.DataStructures.ActionData import ActionData


class CompositeActionFilter(AbsActionFilter):
    """Filter that is a serial composition of several other filters, performing one by one

    Realises the composite OOP pattern
    """

    def __init__(self, filters: list[AbsActionFilter]):
        """

        Args:
            filters (list[AbsActionFilter]): list of filters to sequentially apply
        """

        self.filters = filters
        self.limit = None

    def add_filter(self, filter: AbsActionFilter | list[AbsActionFilter]) -> None:
        """Adds a filter to the composite elements list. Can accept a list
        of filters as a parametre, in this case it will concat
        the registered filters list with the passed filter lidt

        Args:
            filter (AbsActionFilter | list[AbsActionFilter]): a filter or a
            list of filters to add
        """

        if isinstance(filter, list):
            self.filter += filter
        else:
            self.filter.append(filter)

    def filter(self, action_data: ActionData) -> ActionData:
        """Performs the Filter function of all composite elements.
        Filtering is performed one by one - the output of the 1st
        filter is the input of the 2nd filter

        Args:
            action_data (ActionData): list of actions to filter

        Returns:
            ActionData: filtered action list
        """
        actions = action_data.actions
        for filter in self.filters:
            actions = filter.Filter(actions)

        return ActionData(actions=actions)

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """
        return f"Composite action filter. Number of members: {len(self.filter)}"
