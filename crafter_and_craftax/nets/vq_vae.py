from dataclasses import dataclass
from typing import Tuple

from einops import rearrange
from flax import nnx
import jax.numpy as jnp


@dataclass
class EncoderDecoderConfig:
    image_channels: int = 3
    latent_dim: int = 128
    num_channels: int = 64
    mult: Tuple[int] = (1, 1, 2, 2, 4)
    down: Tuple[int] = (1, 0, 1, 1, 0)


class Encoder(nnx.Module):
    def __init__(self, config: EncoderDecoderConfig, rngs: nnx.Rngs) -> None:
        assert len(config.mult) == len(config.down)
        self.config = config
        encoder_layers = [
            nnx.Conv(
                in_features=config.image_channels,
                out_features=config.num_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                rngs=rngs,
            )
        ]
        input_channels = config.num_channels

        for m, d in zip(config.mult, config.down):
            output_channels = m * config.num_channels
            encoder_layers.append(
                ResidualBlock(input_channels, output_channels, rngs=rngs)
            )
            input_channels = output_channels
            if d:
                encoder_layers.append(Downsample(output_channels, rngs=rngs))

        encoder_layers.extend(
            [
                nnx.GroupNorm(num_groups=32, num_features=input_channels, rngs=rngs),
                nnx.relu,
                nnx.Conv(
                    in_features=input_channels,
                    out_features=config.latent_dim,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="SAME",
                    rngs=rngs,
                ),
            ]
        )
        self.encoder = nnx.Sequential(*encoder_layers)

    def __call__(self, x):
        x = self.encoder(x)

        return x


class Decoder(nnx.Module):
    def __init__(self, config: EncoderDecoderConfig, rngs: nnx.Rngs) -> None:
        self.config = config

        assert len(config.mult) == len(config.down)
        decoder_layers = []
        output_channels = config.num_channels

        for m, d in zip(config.mult, config.down):
            input_channels = m * config.num_channels
            decoder_layers.append(
                ResidualBlock(input_channels, output_channels, rngs=rngs)
            )
            output_channels = input_channels
            if d:
                decoder_layers.append(Upsample(input_channels, rngs=rngs))
        decoder_layers.reverse()
        decoder_layers.insert(
            0,
            nnx.Conv(
                in_features=config.latent_dim,
                out_features=input_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                rngs=rngs,
            ),
        )
        decoder_layers.extend(
            [
                nnx.GroupNorm(
                    num_groups=32, num_features=config.num_channels, rngs=rngs
                ),
                nnx.relu,
                nnx.Conv(
                    in_features=config.num_channels,
                    out_features=config.image_channels,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="SAME",
                    rngs=rngs,
                ),
            ]
        )
        self.decoder = nnx.Sequential(*decoder_layers)

    def __call__(self, x):
        x = self.decoder(x)
        return x


class ResidualBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups_norm: int = 32,
        *,
        rngs: nnx.Rngs
    ) -> None:
        self.f = nnx.Sequential(
            nnx.GroupNorm(
                num_groups=num_groups_norm, num_features=in_channels, rngs=rngs
            ),
            nnx.relu,
            nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                rngs=rngs,
            ),
            nnx.GroupNorm(
                num_groups=num_groups_norm, num_features=out_channels, rngs=rngs
            ),
            nnx.relu,
            nnx.Conv(
                in_features=out_channels,
                out_features=out_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                rngs=rngs,
            ),
        )
        self.skip_projection = (
            (lambda x: x)
            if in_channels == out_channels
            else nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(1, 1),
                rngs=rngs,
            )
        )

    def __call__(self, x):
        return self.skip_projection(x) + self.f(x)


class Downsample(nnx.Module):
    def __init__(self, num_channels: int, rngs: nnx.Rngs) -> None:
        self.conv = nnx.Conv(
            in_features=num_channels,
            out_features=num_channels,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x):
        return self.conv(x)


class Upsample(nnx.Module):
    def __init__(self, num_channels: int, rngs: nnx.Rngs) -> None:
        self.conv = nnx.Conv(
            in_features=num_channels,
            out_features=num_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x):
        x = jnp.repeat(x, 2, axis=-3)
        x = jnp.repeat(x, 2, axis=-2)
        return self.conv(x)


class MLPEncoder(nnx.Module):
    def __init__(self, patch_size: int, hidden_dim: int, rngs: nnx.Rngs):
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

class MLPDecoder(nnx.Module):
    def __init__(self, patch_size: int, hidden_dim: int, rngs: nnx.Rngs):
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        self.conv = nnx.ConvTranspose(
            in_features=self.hidden_dim,
            out_features=3,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            rngs=rngs,
        )
    
    def __call__(self, x):
        x = self.conv(x)
        return x