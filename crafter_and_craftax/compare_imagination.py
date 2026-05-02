import jax
import jax.numpy as jnp
from PIL import Image
import pyrallis
import tqdm

from configs import TrainConfig, WorldModelConfig, WorldModelParams
from train import Trainer
from utils.visualization import concat_to_single_image


@pyrallis.wrap()
def test_wm_rollout(cfg: TrainConfig):
    cfg.wandb_config.enable = False
    cfg.wm_config.num_updates = 50
    cfg.ac_config.num_updates = 0
    cfg.total_env_interactions = 10_000
    trainer = Trainer(cfg)
    trainer.restore_state()

    cfg2 = TrainConfig(
        restore_ckpt_path="/storage/username/baseline_our_hyperparams",
        wm_config=WorldModelConfig(
            params=WorldModelParams(
                use_absolute_embedding=False,
                use_spatio_temporal=False,
            ),
            decode_strategy="original",
        ),
    )

    trainer2 = Trainer(cfg2)
    trainer2.restore_state()

    rng = jax.random.PRNGKey(1)
    # Reset environment
    rng, env_rng = jax.random.split(rng)

    trainer.curr_obs, trainer.env_state = trainer.env.reset(env_rng, trainer.env_params)
    trainer.curr_done = jnp.ones((cfg.batch_size,), dtype=jnp.bool)

    # Reset agent state
    trainer.agent_state = trainer.agent.rnn.initialize_carry(cfg.batch_size)

    # # Reset tokenizer
    # trainer.codebook = jnp.zeros((cfg.token_config.params.codebook_size, 7, 7, 3)) - 1
    # trainer.codebook_size = jnp.array(0)

    for _ in tqdm.trange(100):
        rng, rollout_rng = jax.random.split(rng)
        data, next_agent_state = trainer.collect_from_env(rollout_rng, 0)
        trainer.agent_state = next_agent_state

    for j in tqdm.trange(100):

        rng, sample_rng = jax.random.split(rng)
        buffer_data = trainer.buffer.sample(trainer.buffer_state, sample_rng)

        rng, collect_rng = jax.random.split(rng)

        data_wm_baseline = trainer2.collect_from_wm_with_actions(
            buffer_data, collect_rng
        )
        data_wm = trainer.collect_from_wm_with_actions(buffer_data, collect_rng)
        data_wm_ot = trainer.collect_from_wm_with_actions(
            buffer_data, collect_rng, trainer.wm_rollout_with_actions_compare
        )

        real_img = buffer_data.experience["obs"][:, 5:]
        imag_img_baseline = data_wm_baseline[0]
        imag_img = data_wm[0]
        imag_img_ot = data_wm_ot[0]
        for i in range(cfg.batch_size):
            concat_img = jnp.stack(
                (
                    real_img[i][:8],
                    imag_img_baseline[i][:8],
                    imag_img[i][:8],
                    imag_img_ot[i][:8],
                ),
                axis=0,
            )
            concat_img = concat_to_single_image(concat_img, 1)
            concat_img = Image.fromarray(concat_img)
            concat_img.save(f"rollout_comparison/{j}_{i}.png")


if __name__ == "__main__":
    test_wm_rollout()
