import distrax
from flax import nnx
import jax.numpy as jnp


class Actor(nnx.Module):
    def __init__(
        self, input_dim: int, intermediate_dim: int, num_actions: int, *, rngs: nnx.Rngs
    ):
        self.norm1 = nnx.LayerNorm(
            num_features=input_dim,
            rngs=rngs,
        )
        self.linear1 = nnx.Linear(
            in_features=input_dim,
            out_features=intermediate_dim,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=intermediate_dim,
            kernel_init=nnx.initializers.orthogonal(0.01),
            rngs=rngs,
        )
        self.linear3 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=intermediate_dim,
            kernel_init=nnx.initializers.orthogonal(0.01),
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(
            num_features=intermediate_dim,
            rngs=rngs,
        )
        self.linear4 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=num_actions,
            kernel_init=nnx.initializers.orthogonal(0.01),
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.norm1(x)
        x = nnx.relu(self.linear1(x))
        x = x + nnx.relu(self.linear2(x))
        x = x + nnx.relu(self.linear3(x))
        x = self.norm2(x)
        x = self.linear4(x)
        x = distrax.Categorical(logits=x)
        return x


class Critic(nnx.Module):
    def __init__(self, input_dim: int, intermediate_dim: int, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(
            num_features=input_dim,
            rngs=rngs,
        )
        self.linear1 = nnx.Linear(
            in_features=input_dim,
            out_features=intermediate_dim,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=intermediate_dim,
            kernel_init=nnx.initializers.orthogonal(0.01),
            rngs=rngs,
        )
        self.linear3 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=intermediate_dim,
            kernel_init=nnx.initializers.orthogonal(0.01),
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(
            num_features=intermediate_dim,
            rngs=rngs,
        )
        self.linear4 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=1,
            kernel_init=nnx.initializers.orthogonal(1.0),
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.norm1(x)
        x = nnx.relu(self.linear1(x))
        x = x + nnx.relu(self.linear2(x))
        x = x + nnx.relu(self.linear3(x))
        x = self.norm2(x)
        x = self.linear4(x)
        return jnp.squeeze(x, axis=-1)


class ActorCritic(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        *,
        eps: float = 0.2,
        td_loss_coef: float = 2.0,
        ent_loss_coef: float = 0.01,
        rngs: nnx.Rngs
    ):
        intermediate_dim = 2048
        self.eps = eps
        self.td_loss_coef = td_loss_coef
        self.ent_loss_coef = ent_loss_coef

        self.actor = Actor(
            input_dim=input_dim,
            intermediate_dim=intermediate_dim,
            num_actions=num_actions,
            rngs=rngs,
        )
        self.critic = Critic(
            input_dim=input_dim,
            intermediate_dim=intermediate_dim,
            rngs=rngs,
        )

    def __call__(self, state):
        return self.actor(state), self.critic(state)

    def loss(self, state, action, old_pi_log_prob, adv, tgt):
        """
        state: (B, T, 8 * 8 * 128 + 256)
        action: (B, T)
        old_pi_log_prob: (B, T)
        adv: (B, T)
        tgt: (B, T)
        """

        pi, v = self(state)
        log_r = pi.log_prob(action) - old_pi_log_prob

        policy_loss = -jnp.minimum(
            jnp.exp(log_r) * adv,
            jnp.clip(jnp.exp(log_r), 1 - self.eps, 1 + self.eps) * adv,
        ).mean()
        value_loss = jnp.square(v - tgt).mean()
        ent_loss = -pi.entropy().mean()
        total_loss = (
            policy_loss + self.td_loss_coef * value_loss + self.ent_loss_coef * ent_loss
        )

        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "ent_loss": ent_loss,
            "advantage": adv,
            "target": tgt,
        }

        return total_loss, metrics
