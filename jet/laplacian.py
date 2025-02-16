"""Implements a module that computes the Laplacian via jets and can be simplified."""

from typing import Callable, Tuple

from torch import Tensor, eye, zeros
from torch.fx import wrap
from torch.nn import Module

from jet import jet
from jet.utils import replicate, sum_vmapped

# tell `torch.fx` to trace `replicate` as one node (required for simplification)
wrap(replicate)
# tell `torch.fx` to trace `sum_vmapped` as one node (required for simplification)
wrap(sum_vmapped)


class Laplacian(Module):
    """Module that computes the Laplacian of a function using jets."""

    def __init__(
        self, f: Callable[[Tensor], Tensor], dummy_x: Tensor, is_batched: bool
    ):
        """Initialize the Laplacian module.

        Args:
            f: The function whose Laplacian is computed.
            dummy_x: The input on which the Laplacian is computed. It is only used to
                infer meta-data of the function input that `torch.fx` is not capable
                of determining at the moment.
            is_batched: Whether the function and its input are batched. In this case,
                we can use that computations can be carried out independently along
                the leading dimension of tensors.
        """
        super().__init__()
        self.jet_f = jet(f, 2, vmap=True)
        # data that needs to be inferred explicitly from a dummy input
        # because `torch.fx` cannot do this.
        self.x_shape = dummy_x.shape
        self.x_kwargs = {"dtype": dummy_x.dtype, "device": dummy_x.device}

        self.unbatched_dim = (self.x_shape[1:] if is_batched else self.x_shape).numel()
        self.batched_dim = self.x_shape[0] if is_batched else 1
        self.is_batched = is_batched

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the Laplacian of the function at the input tensor.

        Replicates the input tensor, then evaluates the 2-jet of f using
        canonical basis vectors for v1 and zero vectors for v2.

        Args:
            x: Input tensor. Must have same shape as the dummy input tensor that was
               passed in the constructor.

        Returns:
            Tuple containing the replicated function value, the Jacobian, and the
            Laplacian.
        """
        X = replicate(x, self.unbatched_dim)

        V1 = eye(self.unbatched_dim, **self.x_kwargs)
        if self.is_batched:
            V1 = V1.unsqueeze(1).repeat(1, self.batched_dim, 1)
        V1 = V1.reshape(self.unbatched_dim, *self.x_shape)

        V2 = zeros(self.unbatched_dim, *self.x_shape, **self.x_kwargs)

        result = self.jet_f(X, V1, V2)
        return result[0], result[1], sum_vmapped(result[2])
