from flax import nnx
import jax.numpy as jnp


class PatchMLP(nnx.Module):
    def __init__(
        self, patch_size: int, hidden_dim: int, rngs: nnx.Rngs
    ):
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        self.conv1 = nnx.Conv(
            in_features=3,
            out_features=self.hidden_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=self.hidden_dim,
            out_features=self.hidden_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.conv1(x)
        x = nnx.relu(x)
        x = self.conv2(x)

        return x
