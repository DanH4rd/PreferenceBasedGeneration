from src.Abstract.AbsActionFilter import AbsActionFilter
from src.DataStructures.ActionData import ActionData


class EmptyActionFilter(AbsActionFilter):
    """Filter that doest filter and just passes data"""

    def filter(self, action_data: ActionData) -> ActionData:

        return action_data

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """
        return f"Empty filter"
