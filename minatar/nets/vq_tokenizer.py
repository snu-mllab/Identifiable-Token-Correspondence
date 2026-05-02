from dataclasses import dataclass
from typing import Tuple

from einops import rearrange
from flax import nnx
import jax
import jax.numpy as jnp

from nets.vq_vae import Decoder, Encoder, EncoderDecoderConfig


@dataclass
class TokenizerEncoderOutput:
    z: jax.Array  # torch.FloatTensor
    z_quantized: jax.Array  # torch.FloatTensor
    tokens: jax.Array  # torch.LongTensor


class VQTokenizer(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        encoder: Encoder,
        decoder: Decoder,
        obs_shape: Tuple[int, int, int],
        rngs: nnx.Rngs,
    ) -> None:
        self.vocab_size = vocab_size
        self.obs_shape = obs_shape
        enc_dec_config = EncoderDecoderConfig()
        self.encoder = encoder
        self.pre_quant_conv = nnx.Conv(
            in_features=enc_dec_config.latent_dim,
            out_features=embed_dim,
            kernel_size=(1, 1),
            rngs=rngs,
        )
        self.embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=embed_dim,
            embedding_init=nnx.initializers.variance_scaling(
                scale=1.0 / (3.0 * vocab_size), mode="fan_in", distribution="uniform"
            ),
            rngs=rngs,
        )
        self.post_quant_conv = nnx.Conv(
            in_features=embed_dim,
            out_features=enc_dec_config.latent_dim,
            kernel_size=(1, 1),
            rngs=rngs,
        )
        self.decoder = decoder

    def __call__(
        self,
        x,
        should_preprocess: bool = False,
        should_postprocess: bool = False,
    ):
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z + jax.lax.stop_gradient(
            outputs.z_quantized - outputs.z
        )
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions

    @nnx.jit
    def compute_loss(self, obs, **kwargs):
        observations = self.preprocess_input(rearrange(obs, "b t h w c -> (b t) h w c"))
        z, z_quantized, reconstructions = self(
            observations, should_preprocess=False, should_postprocess=False
        )

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        beta = 0.25
        commitment_loss = (
            (jax.lax.stop_gradient(z) - z_quantized) ** 2
        ).mean() + beta * ((z - jax.lax.stop_gradient(z_quantized)) ** 2).mean()

        reconstruction_loss = jnp.abs(observations - reconstructions).mean()

        total_loss = commitment_loss + reconstruction_loss

        metrics = {
            "commitment_loss": commitment_loss,
            "reconstruction_loss": reconstruction_loss,
        }

        return total_loss, metrics

    def encode(self, x, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., H, W, C)
        x = x.reshape(-1, *shape[-3:])
        z = self.encoder(x)
        z = self.pre_quant_conv(z)
        z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)

        b, h, w, e = z.shape
        z_flattened = rearrange(z, "b h w e -> (b h w) e")

        embedding = self.embedding.embedding.value
        embedding = embedding / (
            jnp.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-12
        )

        dist_to_embeddings = (
            jnp.sum(z_flattened**2, axis=1, keepdims=True)
            + jnp.sum(embedding**2, axis=1)
            - 2 * jnp.matmul(z_flattened, jnp.transpose(embedding))
        )

        tokens = dist_to_embeddings.argmin(axis=-1)
        z_q = rearrange(
            self.embedding(tokens), "(b h w) e -> b h w e", b=b, e=e, h=h, w=w
        )
        z_q = z_q / (jnp.linalg.norm(z_q, axis=-1, keepdims=True) + 1e-12)

        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-3], -1)

        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode_from_tokens(self, tokens, should_postprocess: bool = False):
        b, hw = tokens.shape
        h = w = int(hw**0.5)
        z_q = rearrange(self.embedding(tokens), "b (h w) e -> b h w e", h=h, w=w)
        z_q = z_q / (jnp.linalg.norm(z_q, axis=-1, keepdims=True) + 1e-12)
        return self.decode(z_q, should_postprocess)

    def decode(self, z_q, should_postprocess: bool = False):
        shape = z_q.shape  # (..., h, w, E)
        z_q = z_q.reshape(-1, *shape[-3:])
        z_q = self.post_quant_conv(z_q)
        rec = self.decoder(z_q)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        rec = rec[..., : self.obs_shape[0], : self.obs_shape[1], : self.obs_shape[2]]
        return rec

    def encode_decode(
        self,
        x,
        should_preprocess: bool = False,
        should_postprocess: bool = False,
    ):
        z_q = self.encode(x, should_preprocess).z_quantized
        return jax.lax.stop_gradient(self.decode(z_q, should_postprocess))

    def preprocess_input(self, x):
        """x is supposed to be channels first and in [0, 1]"""
        return x * 2 - 1

    def postprocess_output(self, y):
        """y is supposed to be channels first and in [-1, 1]"""
        return (y + 1) / 2.0
