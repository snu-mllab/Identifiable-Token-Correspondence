from math import ceil

import einops
import jax.numpy as jnp


# Modified from https://github.com/lucidrains/improving-transformers-world-model-for-rl/blob/ff0d148a77fc24051ae46d68a1608172b909874c/improving_transformers_world_model/world_model.py#L120
def nonflex_block_causal_mask(seq_len, block_size):
    blocks = ceil(seq_len / block_size)

    causal_mask = jnp.tril(jnp.ones((blocks, blocks), dtype=jnp.bool))
    block_causal_mask = einops.repeat(
        causal_mask, "i j -> (i bsz1) (j bsz2)", bsz1=block_size, bsz2=block_size
    )
    # Add batch and head dimensions
    block_causal_mask = einops.rearrange(block_causal_mask, "q k -> 1 1 q k")
    return block_causal_mask[:, :, :seq_len, :seq_len]
