import logging
from abc import abstractmethod
from typing import Annotated, Any, Generic, Type, TypeVar, Union

from pydantic import BaseModel, BeforeValidator, ValidationInfo

from pram.exception import NoSuitableStrategy

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BuildStrategy(BaseModel, Generic[T]):
    """Use a subset of parameters to build an object."""

    @classmethod
    def matches(cls, parameters: dict) -> bool:
        """Check whether the given parameters contain the keys needed for this strategy."""
        return all(
            k in parameters for k, v in cls.model_fields.items() if v.is_required()
        )

    @abstractmethod
    def build(self) -> T:
        """Build the type from parameters."""
        raise NotImplementedError()


class Builder(Generic[T]):
    """Constructs a type from parameters."""

    def __init__(
        self,
        of_type: Type[T],
        strategies: list[Type[BuildStrategy[T]]],
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

        if isinstance(params, dict):
            for strategy in self._strategies:
                _params = params.copy()

                # Try to populate missing fields, from siblings
                for name in strategy.model_fields:
                    if name not in _params:
                        # Find the value elsewhere in params
                        _params[name] = info.data["params"][name]

                if strategy.matches(_params):
                    return strategy(**_params).build()  # extra=info.data["params"])

            raise NoSuitableStrategy(
                f"No suitable strategy found for keys {', '.join(params.keys())}"
            )

        raise ValueError("???")


def parameterise(type_: Type[T], strategies: list[Type[BuildStrategy[T]]]):
    """Annotates a Type with a set of BuildStrategys."""
    return Annotated[
        type_,
        BeforeValidator(func=Builder(of_type=type_, strategies=strategies)),
    ]
