class LabException(Exception):
    """Lab Exception."""


class NoSuitableStrategy(Exception):
    """No suitable strategy was found for the given parameters."""


class StrategyNoMatch(Exception):
    """The parameters don't match the requirements of the build strategy."""
