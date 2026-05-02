from typing import Optional

import flax
import jax.numpy as jnp
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutputWithPastAndCrossAttentions,
)


@flax.struct.dataclass
class FlaxGPT2WorldModelOutput(FlaxBaseModelOutputWithPastAndCrossAttentions):
    observation_logits: Optional[jnp.ndarray] = None
    reward_logits: Optional[jnp.ndarray] = None
    termination_logits: Optional[jnp.ndarray] = None
