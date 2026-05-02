from dataclasses import asdict
import functools
from flax import nnx
import jax
import jax.numpy as jnp

from nets.impala_cnn import ImpalaCNN
from nets.actor_critic import ActorCritic
from nets.rnn import RNN
from nets.nnt import NearestNeighborTokenizer
from nets.patch_mlp import PatchMLP
from configs import ActorCriticParams


class Agent(nnx.Module):
    def __init__(
        self,
        num_actions: int,
        obs_channels: int = 3,
        ac_params: ActorCriticParams = ActorCriticParams(),
        *,
        rngs: nnx.Rngs
    ):
        self.encoder = ImpalaCNN(
            channels=[64, 64, 128],
            in_channels=obs_channels,
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(
            num_features=2 * 2 * 128,
            rngs=rngs,
        )
        self.linear = nnx.Linear(
            in_features=2 * 2 * 128,
            out_features=256,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            rngs=rngs,
        )
        self.rnn = RNN(
            in_features=256,
            hidden_features=256,
            rngs=rngs,
        )
        self.actor_critic = ActorCritic(
            input_dim=(2 * 2 * 128) + 256,
            num_actions=num_actions,
            **asdict(ac_params),
            rngs=rngs,
        )

    def get_state_from_obs(self, obs, reset, prev_state):
        B, T, *_ = obs.shape
        z = self.encoder(obs.reshape(B * T, *_))
        z = nnx.swish(z)
        z = z.reshape((B, T, -1))

        r_in = self.norm(z)
        r_in = self.linear(r_in)
        r_in = nnx.relu(r_in)

        prev_state, y = self.rnn(prev_state, r_in, reset)
        y = nnx.relu(y)

        state = jnp.concatenate([z, y], axis=-1)

        return state, prev_state

    def __call__(self, obs, reset, prev_state):
        state, prev_state = self.get_state_from_obs(obs, reset, prev_state)

        pi, v = self.actor_critic(state)
        return pi, v, prev_state

    @nnx.jit
    def loss(
        self,
        obs,
        reset,
        prev_state,
        action,
        old_pi_log_prob,
        adv,
        tgt,
        ent_loss_coef,
    ):
        state, prev_state = self.get_state_from_obs(obs, reset, prev_state)

        return self.actor_critic.loss(
            state, action, old_pi_log_prob, adv, tgt, ent_loss_coef
        )


def main():
    model = Agent(num_actions=10, rngs=nnx.Rngs(0))

    x = jnp.ones((16, 20, 63, 63, 3))
    y = model(x)


if __name__ == "__main__":
    main()
