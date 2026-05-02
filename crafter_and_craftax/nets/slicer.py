from typing import Optional

import einops
import flax.linen as nn
import jax.numpy as jnp


def slice_observations(
    state_action_seq: jnp.ndarray, tokens_per_block: int
) -> jnp.ndarray:
    num_blocks = state_action_seq.shape[1] // tokens_per_block

    single_block_indices = jnp.arange(tokens_per_block - 1)[None, :]
    block_offsets = jnp.arange(num_blocks)[:, None] * tokens_per_block
    observation_indices = einops.rearrange(
        block_offsets + single_block_indices, "i j -> (i j)"
    )

    return state_action_seq[:, observation_indices]


class ObservationHead(nn.Module):
    head_module: nn.Module
    tokens_per_block: int

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.head_module(slice_observations(x, self.tokens_per_block))


class ActionHead(nn.Module):
    head_module: nn.Module
    tokens_per_block: int

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        seq_len = x.shape[1]
        num_blocks = seq_len // self.tokens_per_block
        indices = jnp.arange(self.tokens_per_block - 1, seq_len, self.tokens_per_block)
        x = x[:, indices]
        return self.head_module(x)


class Embedder(nn.Module):
    tokens_per_block: int
    max_blocks: int
    observation_embedding: nn.Embed
    action_embedding: nn.Embed
    absolute_embedding: Optional[nn.Embed]

    def setup(self):
        observation_tokens_pattern = jnp.ones(self.tokens_per_block, dtype=jnp.bool)
        observation_tokens_pattern = observation_tokens_pattern.at[-1].set(False)

        self.observation_mask = jnp.tile(observation_tokens_pattern, self.max_blocks)

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        observation_mask = self.observation_mask[: tokens.shape[1]]

        observation_emb = self.observation_embedding(tokens)
        action_emb = self.action_embedding(tokens)

        output = jnp.where(observation_mask[None, :, None], observation_emb, action_emb)

        if self.absolute_embedding is not None:
            absolute_emb = self.absolute_embedding(
                (jnp.arange(tokens.shape[1], dtype=jnp.int32) % self.tokens_per_block)[
                    None, :
                ]
            )
            output += absolute_emb

        return output
