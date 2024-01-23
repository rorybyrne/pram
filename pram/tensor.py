"""Parameterised Tensor"""

import logging
from typing import Annotated, Any, List, Optional, Tuple, Type, TypeVar

import torch
from pydantic import AfterValidator
from torch import Tensor
from torch.nn import Parameter

from pram.exception import StrategyNoMatch
from pram.field import BuildStrategy, parameterise

logger = logging.getLogger(__name__)

T = TypeVar("T")


class GaussianStrategy(BuildStrategy[Tensor]):
    """Sample a Tensor from a Gaussian distribution"""

    mean: float
    std: float
    size: Tuple[int, ...]
    # dale_mask: Optional[Annotated[Tensor, Injection("dale_mask")]]
    train: bool = False
    abs: bool = False
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    density: Optional[Annotated[float, AfterValidator(lambda v: 0 <= v <= 1)]] = None
    round: bool = False
    parity_ratio: Optional[float] = None

    @classmethod
    def parse(cls, parameters: Any, with_context: dict[str, dict]) -> dict[str, Any]:
        if not isinstance(parameters, dict):
            raise StrategyNoMatch()

        if "size" not in parameters:
            parameters["size"] = with_context["local"]["size"]
        parameters["parity_ratio"] = with_context["local"].get("parity_ratio", None)

        if not all(k in parameters for k, v in cls.model_fields.items() if v.is_required()):
            raise StrategyNoMatch()

        return parameters

    def build(self) -> Tensor:
        """Build an instance of Tensor"""
        tensor = torch.normal(mean=self.mean, std=self.std, size=self.size)

        if self.abs:
            tensor = torch.abs(tensor)

        if self.round:
            tensor = tensor.int()

        if self.minimum is not None:
            tensor[tensor < self.minimum] = self.minimum

        if self.maximum is not None:
            tensor[tensor > self.maximum] = self.maximum

        if self.parity_ratio is not None:
            if self.size[0] != self.size[1]:
                raise ValueError("Currently only supporting parity on recurrent tensors")
            n = self.size[0]
            num_exc = int(n * self.parity_ratio)
            # num_inh = n - num_exc

            _temp = torch.ones(n)
            _temp[num_exc:] = -1
            mask = torch.diag(_temp)

        if self.train:
            param = Parameter(tensor, requires_grad=True)

            return param
        return tensor


class NumberStrategy(BuildStrategy[Tensor]):
    value: float
    size: Tuple[int, ...]

    @classmethod
    def parse(cls, parameters: Any, with_context: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(parameters, (float, int)):
            raise StrategyNoMatch()

        return {"value": float(parameters), "size": with_context["local"]["size"]}

    def build(self) -> Tensor:
        return torch.full(self.size, self.value)


default_strategies: List[Type[BuildStrategy[Tensor]]] = [GaussianStrategy, NumberStrategy]


ParameterisedTensor = parameterise(Tensor, default_strategies)
