import json
from nets.world_model import get_default_position_ids
import pyrallis

import jax
import jax.numpy as jnp
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
import orbax.checkpoint as ocp
from PIL import Image

from configs import TrainConfig
from train import Trainer
from utils.visualization import concat_to_single_image


@pyrallis.wrap()
def main(cfg: TrainConfig):

    trainer = Trainer(cfg)

    trainer.restore_state()

    buffer = trainer.buffer_state

    rng = jax.random.PRNGKey(cfg.dataset_seed)

    action_count = [0 for _ in range(17)]
    dataset = []

    n_data = 1000
    curr = 0
    while curr < n_data:
        print(f"{action_count=}")
        rng, sample_rng = jax.random.split(rng)
        data = trainer.buffer.sample(trainer.buffer_state, sample_rng)

        obs = data["experience"]["obs"]
        action = data["experience"]["action"]

        token = trainer.tokenizer(obs, trainer.codebook)

        state_action_ids = jnp.concatenate(
            (
                token[:, 0],
                action[:, 0][:, None],
            ),
            axis=-1,
        )
        position_ids = get_default_position_ids(token.shape[0], 82, 82, True)

        past_key_values = trainer.world_model.init_cache(token.shape[0], 82 * 20)

        outputs = trainer.world_model(
            trainer.world_model_train_state.params,
            state_action_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        next_state_logits = outputs.observation_logits[:, -81:]

        rng, state_rng = jax.random.split(rng)

        mask_codebook = (jnp.arange(4096) >= trainer.codebook_size) * (-jnp.inf)
        next_state_logits = next_state_logits + mask_codebook
        next_state_logits = next_state_logits + jax.random.gumbel(
            state_rng, next_state_logits.shape
        )
        next_state_ids = jnp.argmax(next_state_logits, axis=-1)

        curr += token.shape[0]

        try:
            for i in range(token.shape[0]):
                dataset.append(
                    {
                        "curr_obs": obs[i, 0],
                        "curr_token": token[i, 0],
                        "action": action[i, 0],
                        "next_obs": obs[i, 1],
                        "next_token": token[i, 1],
                        "transformer_logits": next_state_logits[i],
                        "transformer_preds": next_state_ids[i],
                    }
                )
                action_count[action[i, 0]] += 1
        except:
            pass

    # TODO: Set your chosen dataset save directory
    with ocp.CheckpointManager(f"dataset_dir_{cfg.dataset_seed}") as ckpt_mngr:
        ckpt_mngr.save(
            0,
            args=ocp.args.Composite(
                curr_obs=ocp.args.ArraySave(
                    jnp.array([data["curr_obs"] for data in dataset])
                ),
                curr_token=ocp.args.ArraySave(
                    jnp.array([data["curr_token"] for data in dataset])
                ),
                action=ocp.args.ArraySave(
                    jnp.array([data["action"] for data in dataset])
                ),
                next_obs=ocp.args.ArraySave(
                    jnp.array([data["next_obs"] for data in dataset])
                ),
                next_token=ocp.args.ArraySave(
                    jnp.array([data["next_token"] for data in dataset])
                ),
                transformer_logits=ocp.args.ArraySave(
                    jnp.array([data["transformer_logits"] for data in dataset])
                ),
                transformer_preds=ocp.args.ArraySave(
                    jnp.array([data["transformer_preds"] for data in dataset])
                ),
            ),
        )


if __name__ == "__main__":
    main()
