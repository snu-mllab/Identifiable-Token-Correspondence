import numpy as np


def concat_to_single_image(obs, sep=0):
    ndim = len(obs.shape)

    if ndim == 3:  # H, W, C
        obs = obs[None, None]
    elif ndim == 4:  # T, H, W, C
        obs = obs[None]

    obs = np.pad(obs, ((0, 0), (0, 0), (0, sep), (0, sep), (0, 0)))
    B, T, H, W, C = obs.shape

    obs = obs.transpose(0, 2, 1, 3, 4)
    obs = obs.reshape(B * H, T * W, C)

    obs_np = np.array(obs * 255).astype(np.uint8)

    return obs_np
