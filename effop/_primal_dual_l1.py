import logging
from typing import Optional

import torch
from torch import Tensor

from ._finite_diff_operator import FiniteDifferenceL1
from ._linop import LinearOperator


class PrimalDualL1:
    """
    Primal-dual L2/L1 solver.

    This implements the monotone FISTA algorithm of Beck/Teboulle, which is a
    Nesterov momentum-accelerated variant of the primal-dual algorithm.

    Note that this class does not actually implement the monotone checks, so
    the user expected to set the ```dual_iterations``` parameter high enough
    such that the non-monotonicity of FISTA is not a major concern.

    The implementation is based on the MATLAB BARISTA code available at
    https://github.com/mmuckley/MRI_FISTA/blob/master/matlab/ir_barista_analysis.m
    However, the present code removes all non-scalar step sizes and adaptive
    restart mechanisms for simplicity of presentation.

    Args:
        data_operator: Linear operator for data term.
        data_bound: Bound on the Lipschitz constant for the data operator.
            Typically, this is the maximum eigenvalue of A'A.
        reg_operator: Linear operator for L1 term.
        reg_bound: Bound on maximum eigenvalue of T'T.
        num_iterations: Number of out primal-dual iterations to run.
        data_weights: Data fidelity weights. For MRI, these are the density
            compensation weights.
        dual_iterations: Number of iterations for the dual inner problem to
            run. For MRI, these are much cheaper than outer iterations, so a
            high number is acceptable.
    """

    def __init__(
        self,
        data_operator: LinearOperator,
        data_bound: float,
        reg_operator: FiniteDifferenceL1,
        reg_bound: float,
        num_iterations: int,
        data_weights: Optional[Tensor] = None,
        dual_iterations: int = 20,
    ):
        self.data_operator = data_operator
        self.data_step = 1.0 / data_bound
        self.reg_operator = reg_operator
        self.reg_step = 1.0 / reg_bound
        self.num_iterations = num_iterations
        if data_weights is None:
            self.data_weights = torch.tensor(1.0)
        else:
            self.data_weights = data_weights
        self.dual_iterations = dual_iterations

        self.logger = logging.getLogger(self.__class__.__name__)

    def _calculate_objective(
        self, est: Tensor, data: Tensor, data_est: Tensor, weights: Tensor
    ) -> float:
        data_term = 0.5 * (weights * (data - data_est).abs() ** 2).sum()
        reg_term = self.reg_operator.cost(est)

        print(f"data: {data_term}, reg: {reg_term}")
        return float(data_term + reg_term)

    def _complex_inprod(self, a: Tensor, b: Tensor) -> Tensor:
        return (a.conj() * b).sum()

    def solve(self, data: Tensor, x: Tensor) -> Tensor:
        weights = self.data_weights.to(x.device)
        if x.dtype == torch.complex128:
            weights = weights.to(torch.float64)
        else:
            weights = weights.to(torch.float32)
        z = x
        lam = self.reg_operator.lam
        betainv = 1.0 / self.reg_operator.lam
        tau = torch.tensor(1.0)
        q = betainv * self.reg_step * self.reg_operator.forward(x)
        q[q.abs() > 1.0] = torch.sgn(q[q.abs() > 1.0])

        for ind in range(self.num_iterations):
            # gradients
            Az = self.data_operator.forward(z)
            cost = self._calculate_objective(x, data, Az, weights)
            print(f"ite {ind-1}: cost: {cost}")
            ngrad = self.data_operator.adjoint(weights * (data - Az))

            oldest = x
            oldtau = tau
            tau = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 + tau**2))

            b = z + self.data_step * ngrad

            # run dual iterations
            innertau = torch.tensor(1.0)
            w = q
            x = b - self.data_step * (self.reg_operator.adjoint(lam * w))
            for _ in range(self.dual_iterations):
                qold = q
                oldinnertau = innertau
                innertau = 0.5 * (1 + torch.sqrt(1.0 + 4.0 * innertau**2))

                ngrad = self.reg_operator.forward(x)

                q = w + betainv * (self.reg_step * ngrad)

                # project on to convex set
                q[q.abs() > 1.0] = torch.sgn(q[q.abs() > 1.0])

                # update momentum variable
                qdiff = q - qold
                w = q + ((oldinnertau - 1) / innertau) * qdiff

                x = b - self.data_step * (self.reg_operator.adjoint(lam * w))

            x = b - self.data_step * self.reg_operator.adjoint(lam * q)
            diffest = x - oldest

            z = x + ((oldtau - 1.0) / tau) * diffest

        return x
