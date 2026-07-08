from src.Abstract.AbsActionFilter import AbsActionFilter
from src.DataStructures.ActionData import ActionData


class CompositeActionFilter(AbsActionFilter):
    """Filter that is a serial composition of several other filters, performing one by one

    Realises the composite OOP pattern
    """

    def __init__(self, filters: list[AbsActionFilter] = []):
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
            self.filters += filter
        else:
            self.filters.append(filter)

    def filter(self, action_data: ActionData) -> ActionData:
        """Performs the Filter function of all composite elements.
        Filtering is performed one by one - the output of the 1st
        filter is the input of the 2nd filter

        Args:
            action_data (ActionData): list of actions to filter

        Returns:
            ActionData: filtered action list
        """

        if self.is_empty():
            raise Exception("No filters are present in composite series filter")

        for series_filter in self.filters:
            action_data = series_filter.filter(action_data)

        return action_data

    def is_empty(self):
        return len(self.filters) == 0

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """
        return f"Composite action filter. Number of members: {len(self.filters)}"
