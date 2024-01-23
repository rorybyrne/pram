from __future__ import annotations

import logging
from abc import abstractmethod
from copy import deepcopy
from pprint import pprint
from typing import Any, Generic, List, Type, TypeVar, Union

from pydantic import BaseModel, BeforeValidator, ValidationInfo
from typing_extensions import Annotated

from pram.exception import NoSuitableStrategy, StrategyNoMatch

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BuildStrategy(BaseModel, Generic[T]):
    """Use a subset of parameters to build an object."""

    @classmethod
    def parse(cls, parameters: Any, with_context: dict[str, Any]) -> dict[str, Any]:
        ...

    @abstractmethod
    def build(self) -> T:
        """Build the type from parameters."""
        raise NotImplementedError()


class Builder(Generic[T]):
    """Constructs a type from parameters."""

    def __init__(
        self,
        of_type: Type[T],
        strategies: List[Type[BuildStrategy[T]]],
    ) -> None:
        self._type = of_type
        self._strategies = strategies

    def __call__(self, params: Union[dict, T], info: ValidationInfo) -> Any:
        """Construct a field from parameters.

        This is called for each field on the model.
        """
        if isinstance(params, self._type):
            # incase the user wants to build a model directly
            return params

        assert info.context is not None, "Must pass context. Library bug."

        for strategy in self._strategies:
            _params = deepcopy(params)
            try:
                # info.data contains the other processed local fields
                _params = strategy.parse(_params, with_context={"local": info.data, "global": info.context})
            except StrategyNoMatch:
                continue

            return strategy(**_params).build()

        raise NoSuitableStrategy(f"No suitable strategy found for keys {', '.join(params.keys())}")


def parameterise(type_: Type[T], strategies: List[Type[BuildStrategy[T]]]):
    """Annotates a Type with a set of BuildStrategys."""
    return Annotated[
        type_, BeforeValidator(func=Builder(of_type=type_, strategies=strategies))
    ]  # , BeforeValidator(func=inject_params)
