import jax.numpy as jnp
from flax import nnx


class RNN(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int, *, rngs: nnx.Rngs):
        self.hidden_features = hidden_features
        self.rnn_cell = nnx.GRUCell(
            in_features=in_features,
            hidden_features=hidden_features,
            rngs=rngs,
        )

    def __call__(self, prev_state, z, reset):
        def single_step(prev_state, z, reset):
            prev_state = jnp.where(
                reset[:, None], jnp.zeros_like(prev_state), prev_state
            )
            return self.rnn_cell(prev_state, z)

        prev_state, y = nnx.scan(
            single_step,
            in_axes=(nnx.transforms.iteration.Carry, 1, 1),
            out_axes=(nnx.transforms.iteration.Carry, 1),
        )(prev_state, z, reset)

        return prev_state, y

    def initialize_carry(self, batch_size):
        return self.rnn_cell.initialize_carry((batch_size, self.hidden_features))
