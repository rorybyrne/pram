"""Parameterised Tensor"""

import logging
from typing import List, Tuple, Type, TypeVar

import torch
from torch import Tensor

from pram.field import BuildStrategy, parameterise

logger = logging.getLogger(__name__)

T = TypeVar("T")


class GaussianStrategy(BuildStrategy[Tensor]):
    """Sample a Tensor from a Gaussian distribution"""

    mean: float
    std: float
    size: Tuple[int, int]

    def build(self) -> Tensor:
        """Build an instance of Tensor"""
        return torch.normal(**self.model_dump())


default_strategies: List[Type[BuildStrategy[Tensor]]] = [GaussianStrategy]


ParameterisedTensor = parameterise(Tensor, default_strategies)
