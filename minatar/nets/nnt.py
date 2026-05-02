import functools
import math
from typing import Optional

from einops import rearrange
from flax import nnx
import jax.numpy as jnp


class NearestNeighborTokenizer:
    def __init__(
        self,
        codebook_size: int = 4096,
        patch_size: int = 7,
        patch_channels: int = 3,
        grid_row: int = 9,
        grid_col: int = 9,
        threshold: Optional[float] = 0.75,
    ):
        self.codebook_size = codebook_size
        self.patch_size = patch_size
        self.patch_channels = patch_channels
        self.grid_row = grid_row
        self.grid_col = grid_col
        self.threshold = threshold

    def init_codebook(self):
        return (
            jnp.zeros(
                (
                    self.codebook_size,
                    self.patch_size,
                    self.patch_size,
                    self.patch_channels,
                )
            )
            - 1
        )

    @staticmethod
    def update_codebook(codebook_info, patch, threshold, max_size):
        codebook, current_size = codebook_info

        if threshold is None:
            matches = jnp.isclose(codebook, patch).all(axis=(-3, -2, -1))
            has_match = matches.any()
            should_update = jnp.logical_and(~has_match, current_size < max_size)
        else:
            diff = codebook - patch
            diff = jnp.square(diff).sum(axis=(-3, -2, -1))
            should_update = jnp.logical_and(
                (diff > threshold).all(), current_size < max_size
            )

        codebook = jnp.where(
            should_update, codebook.at[current_size].set(patch), codebook
        )
        current_size += should_update

        return (codebook, current_size), None

    @functools.partial(nnx.jit, static_argnums=(0,))
    def update(self, x, codebook, codebook_size):
        *_, H, W, C = x.shape

        x = x.reshape(
            -1, self.grid_row, self.patch_size, self.grid_col, self.patch_size, C
        )
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, self.patch_size, self.patch_size, C)

        (codebook, codebook_size), _ = nnx.scan(
            self.update_codebook,
            in_axes=(nnx.transforms.iteration.Carry, 0, None, None),
        )((codebook, codebook_size), x, self.threshold, self.codebook_size)

        return codebook, codebook_size

    @functools.partial(nnx.jit, static_argnums=(0,))
    def __call__(self, x, codebook):
        *_, H, W, C = x.shape

        x = x.reshape(
            -1, self.grid_row, self.patch_size, self.grid_col, self.patch_size, C
        )
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, self.patch_size, self.patch_size, C)
        diff = x[:, None] - codebook[None]
        diff = jnp.square(diff).sum(axis=(-3, -2, -1))
        idx = jnp.argmin(diff, axis=-1)

        idx = idx.reshape(*_, self.grid_row * self.grid_col)

        return idx

    @functools.partial(nnx.jit, static_argnums=(0,))
    def decode(self, x, codebook):
        *_, seq_len = x.shape
        x = x.reshape(-1, self.grid_row, self.grid_col)
        x = jnp.take(codebook, x, axis=0)
        x = jnp.clip(x, 0.0, 1.0)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(
            *_,
            self.grid_row * self.patch_size,
            self.grid_col * self.patch_size,
            x.shape[-1],
        )

        return x


class SubpatchTokenizer(NearestNeighborTokenizer):
    def __init__(
        self,
        codebook_size: int = 4096,
        patch_size: int = 7,
        threshold: Optional[float] = None,
        subpatch_size: int = 4,
    ):
        super().__init__(codebook_size, patch_size, threshold)
        self.subpatch_size = subpatch_size

    def init_codebook(self):
        return (
            jnp.zeros(
                (
                    self.codebook_size,
                    self.subpatch_size,
                    self.subpatch_size,
                    3,
                )
            )
            - 1
        )

    @functools.partial(nnx.jit, static_argnums=(0,))
    def update(self, x, codebook, codebook_size):
        *Bs, H, W, C = x.shape

        x = rearrange(
            x,
            "... (hg hp) (wg wp) c -> (...) hg wg hp wp c",
            hp=self.patch_size,
            wp=self.patch_size,
        )

        s = self.subpatch_size
        top_left = x[:, :, :, :s, :s]
        top_right = x[:, :, :, :s, -s:]
        bottom_left = x[:, :, :, -s:, :s]
        bottom_right = x[:, :, :, -s:, -s:]

        subpatches = jnp.stack(
            [top_left, top_right, bottom_left, bottom_right], axis=-4
        )

        subpatches = rearrange(subpatches, "b hg wg s hp wp c -> (b hg wg s) hp wp c")

        (codebook, codebook_size), _ = nnx.scan(
            self.update_codebook,
            in_axes=(nnx.transforms.iteration.Carry, 0, None, None),
        )((codebook, codebook_size), subpatches, self.threshold, self.codebook_size)

        return codebook, codebook_size


class SuperpatchTokenizer(NearestNeighborTokenizer):
    def __init__(
        self,
        codebook_size: int = 4096,
        patch_size: int = 7,
        threshold: Optional[float] = None,
        superpatch_height: int = 2,
        superpatch_width: int = 2,
    ):
        super().__init__(codebook_size, patch_size, threshold)

        assert superpatch_height in [1, 2]
        assert superpatch_width in [1, 2]
        self.superpatch_height = superpatch_height
        self.superpatch_width = superpatch_width
        self.supergrid_height = math.ceil(self.grid_size / superpatch_height)
        self.supergrid_width = math.ceil(self.grid_size / superpatch_width)
        self.screen_size = self.grid_size * self.patch_size

    def init_codebook(self):
        return (
            jnp.zeros(
                (
                    self.codebook_size,
                    self.superpatch_height * self.patch_size,
                    self.superpatch_width * self.patch_size,
                    3,
                )
            )
            - 1
        )

    def _rearrange_to_patches(self, x):
        x = rearrange(
            x,
            "... (hg hp) (wg wp) c -> (...) hg wg hp wp c",
            hp=self.patch_size,
            wp=self.patch_size,
        )

        repeats = jnp.ones(self.grid_size, dtype=jnp.uint8).at[-1].set(2)
        if self.superpatch_height == 2:
            # Make grid y-axis even by repeating last row
            x = jnp.repeat(x, repeats, axis=-5, total_repeat_length=self.grid_size + 1)
        if self.superpatch_width == 2:
            # Make grid x-axis even by repeating last column
            x = jnp.repeat(x, repeats, axis=-4, total_repeat_length=self.grid_size + 1)

        x = rearrange(
            x,
            "b (h_supergrid h_subpatch) (w_supergrid w_subpatch) hp wp c -> (b h_supergrid w_supergrid) (h_subpatch hp) (w_subpatch wp) c",
            h_subpatch=self.superpatch_height,
            w_subpatch=self.superpatch_width,
        )

        return x

    @functools.partial(nnx.jit, static_argnums=(0,))
    def update(self, x, codebook, codebook_size):
        x = self._rearrange_to_patches(x)

        (codebook, codebook_size), _ = nnx.scan(
            self.update_codebook,
            in_axes=(nnx.transforms.iteration.Carry, 0, None, None),
        )((codebook, codebook_size), x, self.threshold, self.codebook_size)

        return codebook, codebook_size

    @functools.partial(nnx.jit, static_argnums=(0,))
    def __call__(self, x, codebook):
        *Bs, H, W, C = x.shape
        x = self._rearrange_to_patches(x)

        diff = x[:, None] - codebook[None]
        diff = jnp.square(diff).sum(axis=(-3, -2, -1))
        idx = jnp.argmin(diff, axis=-1)

        idx = idx.reshape(*Bs, self.supergrid_height * self.supergrid_width)

        return idx

    @functools.partial(nnx.jit, static_argnums=(0,))
    def decode(self, x, codebook):
        *Bs, seq_len = x.shape
        x = x.reshape(-1, self.supergrid_height, self.supergrid_width)
        x = jnp.take(codebook, x, axis=0)
        x = jnp.clip(x, 0.0, 1.0)

        x = rearrange(x, "b hsg wsg hsp wsp c -> b (hsg hsp) (wsg wsp) c")

        # Crop away duplicated tile
        x = x[:, : self.screen_size, : self.screen_size]

        x = x.reshape(*Bs, *x.shape[1:])

        return x
