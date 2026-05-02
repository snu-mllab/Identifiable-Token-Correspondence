import einops
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from PIL import Image
from utils.visualization import concat_to_single_image


def restore_state(restore_ckpt_path, restore_ckpt_step=None):
    with ocp.CheckpointManager(restore_ckpt_path) as ckpt_mngr:
        step = restore_ckpt_step
        if step is None:
            step = ckpt_mngr.latest_step()

        restored = ckpt_mngr.restore(
            step,
            args=ocp.args.Composite(
                codebook=ocp.args.ArrayRestore(),
                codebook_size=ocp.args.ArrayRestore(),
                # tgt_mean=ocp.args.ArrayRestore(),
                # tgt_std=ocp.args.ArrayRestore(),
                # debiasing=ocp.args.ArrayRestore(),
                # buffer_state=ocp.args.StandardRestore(),
                policy_train_state=ocp.args.StandardRestore(),
                world_model_train_state=ocp.args.StandardRestore(),
            ),
        )
        codebook = restored.codebook
        codebook_size = restored.codebook_size

    return codebook, codebook_size


if __name__ == "__main__":
    # TODO: Change to your saved model checkpoint directory
    codebook, codebook_size = restore_state("model_checkpoint_dir")

    codebook = codebook[:500]

    code_diff = codebook[:, None] - codebook[None, :]
    code_diff = (code_diff**2).sum(axis=(-3, -2, -1))

    codebook = codebook.reshape(25, 20, 7, 7, 3)

    img = Image.fromarray(concat_to_single_image(codebook, sep=1))
    img.save("codebook.png")
