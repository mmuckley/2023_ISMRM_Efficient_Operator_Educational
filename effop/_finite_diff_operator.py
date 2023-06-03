import torch
from torch import Tensor

from ._linop import LinearOperator


class FiniteDifferenceL1(LinearOperator):
    """
    Finite difference operator.

    Effectively applies image[1:] - image[:-1] and adjoint.

    Args:
        lam: regularization parameter (for cost() method)
    """
    def __init__(self, lam: float):
        super().__init__()
        self.lam = lam

    def forward(self, image: Tensor) -> Tensor:
        # assume input is of size (num_timepoints, num_coils, ny, nx)
        return image[1:] - image[:-1]

    def adjoint(self, diffs: Tensor) -> Tensor:
        # assume input is of size (num_timepoints-1, num_coils, ny, nx)
        return torch.cat(
            (diffs[0].unsqueeze(0) * -1, diffs[:-1] - diffs[1:], diffs[-1].unsqueeze(0))
        )

    def cost(self, image: Tensor) -> Tensor:
        return self.lam * self.forward(image).abs().sum()


class FiniteDifferenceWithHyperbola(LinearOperator):
    def __init__(self, lam: float, smooth_param: float = 1e-15):
        super().__init__()
        self._lam = lam
        self._smooth_param = smooth_param

    def forward(self, image: Tensor) -> Tensor:
        # assume input is of size (num_timepoints, num_coils, ny, nx)
        return image[1:] - image[:-1]

    def adjoint(self, diffs: Tensor) -> Tensor:
        # assume input is of size (num_timepoints-1, num_coils, ny, nx)
        return torch.cat(
            (diffs[0].unsqueeze(0) * -1, diffs[:-1] - diffs[1:], diffs[-1].unsqueeze(0))
        )

    def cost(self, image: Tensor) -> Tensor:
        return (
            self._lam
            * (self.forward(image).abs() ** 2 + self._smooth_param).sqrt().sum()
        )

    def _wpot(self, coefs: Tensor) -> Tensor:
        return torch.rsqrt(1 + coefs.abs() ** 2 / self._smooth_param)

    def reg_grad(self, est: Tensor) -> Tensor:
        coefs = self.forward(est)
        return self.adjoint(self._lam * self._wpot(coefs) * coefs)

    def _abs_forw(self, image: Tensor) -> Tensor:
        return image[1:] + image[:-1]

    def _abs_back(self, diffs: Tensor) -> Tensor:
        return torch.cat(
            (
                diffs[0].unsqueeze(0) * 1.0,
                diffs[:-1] + diffs[1:],
                diffs[-1].unsqueeze(0),
            )
        )

    def reg_denom(self, est: Tensor) -> Tensor:
        d = self.forward(est)
        ck = self._abs_forw(torch.ones_like(est).abs())
        wt = self._lam * self._wpot(d)
        return self._abs_back(wt * ck)
