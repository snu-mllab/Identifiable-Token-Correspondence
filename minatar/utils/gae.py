from flax import nnx
import jax.numpy as jnp

def calc_adv_tgt(reward, done, old_value, gamma, ld):
    delta = reward + (1 - done) * gamma * old_value[:, 1:] - old_value[:, :-1]

    def calc_adv(next_adv, dt):
        adv = gamma * ld * next_adv + dt
        return adv, adv

    _, adv = nnx.scan(
        calc_adv,
        in_axes=(nnx.transforms.iteration.Carry, 1),
        out_axes=(nnx.transforms.iteration.Carry, 1),
        reverse=True,
    )(jnp.zeros_like(delta[:, 0]), delta)

    adv = (1 - done) * adv + done * delta

    tgt = adv + old_value[:, :-1]

    return adv, tgt