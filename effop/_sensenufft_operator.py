from typing import Optional

from torch import Tensor
from torchkbnufft import KbNufft, KbNufftAdjoint

from ._linop import LinearOperator


class SenseNufftOp(LinearOperator):
    """
    MRI Non-Cartesian SENSE operator with NUFFT.

    This implements FSx, where F is a Non-Cartesian NUFFT operator and S is an
    MRI sensistivty operator.

    Args:
        sensitivity_maps: A (C, H, W)-size tensor with sensitivity maps
            (C coils).
        ktraj: Theh k-space trajecotry (ndim, klength).
        dcomp: The density compensation weights. These are only used for the
            adjoint. Note: the solver for this package can also consider data
            weights. Do not use the weights here and in the solver
            simultaneously.
    """

    _sensitivity_maps: Tensor
    _ktraj: Tensor
    _dcomp: Optional[Tensor]

    def __init__(
        self, sensitivity_maps: Tensor, ktraj: Tensor, dcomp: Optional[Tensor] = None
    ):
        super().__init__()
        self.register_buffer("_sensitivity_maps", sensitivity_maps)
        self.register_buffer("_ktraj", ktraj)
        if dcomp is None:
            self._dcomp = None
        else:
            self.register_buffer("_dcomp", dcomp)

        im_size = (sensitivity_maps.shape[-2], sensitivity_maps.shape[-1])
        self._kbnufft = KbNufft(im_size=im_size)
        self._kbnufftadjoint = KbNufftAdjoint(im_size=im_size)

    def forward(self, image: Tensor) -> Tensor:
        # assume input is (num_timepoints, num_coils, ny, nx)
        return self._kbnufft(
            image, self._ktraj, smaps=self._sensitivity_maps, norm="ortho"
        )

    def adjoint(self, data: Tensor) -> Tensor:
        # assume input is (num_timepoints, num_coils, num_kspace)
        if self._dcomp is not None:
            data = self._dcomp * data
        return self._kbnufftadjoint(
            data, self._ktraj, smaps=self._sensitivity_maps, norm="ortho"
        )
