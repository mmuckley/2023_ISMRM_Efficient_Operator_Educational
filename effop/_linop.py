from typing import List

import torch.nn as nn


class LinearOperator(nn.Module):
    """
    Abstract base class for linear opeartors.

    Implementer is expected to subclass this object and overwrite forward()
    and adjoint() methods.
    """
    def forward(self, y):
        raise NotImplementedError

    def adjoint(self, y):
        raise NotImplementedError

    def reg_denom(self, x):
        raise NotImplementedError


class ChainedLinearOperator(LinearOperator):
    """
    Chained linear operator.

    This is effectively a wrapper class for chaining multiple linear operators
    together.

    Args:
        linops: list of LinearOperator objects.
    """
    def __init__(self, linops: List[LinearOperator]):
        super().__init__()
        self._linops = linops

    def forward(self, x):
        for linop in self._linops:
            x = linop.forward(x)

        return x

    def adjoint(self, y):
        for linop in reversed(self._linops):
            y = linop.adjoint(y)

        return y
