"""Implements Laplacians via jets."""

from torch import eye, zeros
from torch.fx import wrap
from torch.nn import Module

from jet import jet
from jet.utils import replicate, sum_vmapped

# tell `torch.fx` to trace `replicate` as one node
wrap(replicate)
# tell `torch.fx` to trace `sum_vmapped` as one node
wrap(sum_vmapped)


class Laplacian(Module):
    def __init__(self, f, dummy_x):
        super().__init__()
        self.jet_f = jet(f, 2, vmap=True)
        # data that needs to be inferred explicitly from a dummy input
        # because `torch.fx` cannot do this.
        self.x_numel = dummy_x.numel()
        self.x_shape = dummy_x.shape
        self.x_dtype = dummy_x.dtype
        self.x_device = dummy_x.device

    def forward(self, x):
        X = replicate(x, self.x_numel)
        V1 = eye(self.x_numel, dtype=self.x_dtype, device=self.x_device).reshape(
            self.x_numel, *self.x_shape
        )
        V2 = zeros(
            self.x_numel, *self.x_shape, dtype=self.x_dtype, device=self.x_device
        )
        result = self.jet_f(X, V1, V2)
        return result[0], result[1], sum_vmapped(result[2])
