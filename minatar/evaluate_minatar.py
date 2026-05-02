import os

from dataclasses import asdict
import pprint
import pyrallis
import tqdm
import wandb

import jax
import jax.numpy as jnp

from configs import TrainConfig
from train import Trainer


@pyrallis.wrap()
def main(cfg: TrainConfig):
    pprint.pprint(cfg)

    if cfg.wandb_config.enable:
        wandb.init(
            project=cfg.wandb_config.project_name,
            config=asdict(cfg),
            name=f"{cfg.wandb_config.exp_name}_s{cfg.seed}",
            group=cfg.wandb_config.group_name,
        )

    trainer = Trainer(cfg)

    trainer.restore_state()

    # Reset environment
    rng, env_rng = jax.random.split(trainer.rng)

    trainer.curr_obs, trainer.env_state = trainer.env.reset(env_rng, trainer.env_params)
    trainer.curr_done = jnp.ones((cfg.batch_size,), dtype=jnp.bool)

    # Reset agent state
    trainer.agent_state = trainer.agent.rnn.initialize_carry(cfg.batch_size)

    sum_returns = jnp.zeros((), dtype=jnp.float32)
    sum_ends = jnp.zeros((), dtype=jnp.uint32)

    step = 0
    while (sum_ends < 1000).any():
        rng, rollout_rng = jax.random.split(rng)

        data, next_agent_state, info = trainer.collect_from_env(rollout_rng, 0)

        sum_returns += jnp.sum(
            info["returned_episode_returns"],
            where=info["returned_episode"],
        )
        sum_ends += jnp.sum(info["returned_episode"])

        trainer.agent_state = next_agent_state
        step += cfg.rollout_horizon

        print(f"# of episodes finished: {sum_ends}")

    print(f"Average return over {sum_ends} episodes: {sum_returns / sum_ends}")

    filename = f"{cfg.wandb_config.exp_name}/{cfg.env_config.env_name}/trash_cost_{cfg.wm_config.trash_cost}/seed{cfg.seed}/{cfg.restore_ckpt_step}.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(f"{sum_returns / sum_ends}")


if __name__ == "__main__":
    main()
