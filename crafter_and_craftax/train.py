from dataclasses import asdict
import functools
import math
import pprint
import time

from craftax import craftax_env
import flashbax as fbx
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flax import nnx
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from ott.geometry.geometry import Geometry
from ott.solvers import linear
from ott.tools.unreg import hungarian
import pyrallis
from tqdm import tqdm
import wandb

from configs import TrainConfig
from env.wrapper import (
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    LogWrapper,
)
from nets.agent import Agent
from nets.nnt import NearestNeighborTokenizer, SuperpatchTokenizer
from nets.configuration import GPT2WorldModelConfig
from nets.world_model import FlaxGPT2WorldModelModule, get_default_position_ids
from utils.gae import calc_adv_tgt
from utils.visualization import concat_to_single_image


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

        self.multi_gpu = len(jax.devices()) > 1

        if self.cfg.ckpt_path is not None:
            self.cfg.ckpt_path = self.cfg.ckpt_path.format(cfg=self.cfg)
            self.cfg.ckpt_path = f"{self.cfg.ckpt_path}/fixed_step_seed{self.cfg.seed}"

        if self.cfg.restore_ckpt_path is not None:
            self.cfg.restore_ckpt_path = self.cfg.restore_ckpt_path.format(cfg=self.cfg)
            self.cfg.restore_ckpt_path = (
                f"{self.cfg.restore_ckpt_path}/fixed_step_seed{self.cfg.seed}"
            )

        self.rng = jax.random.PRNGKey(cfg.seed)
        self.env, self.env_params, self.num_actions, self.Achievement, self.Action = (
            self.build_environment(cfg.batch_size)
        )
        self.rollout = self.build_rollout(self.env, self.env_params)

        self.agent, self.policy_train_state = self.build_agent()

        self.rng, wm_rng = jax.random.split(self.rng)
        (
            self.world_model,
            self.world_model_params,
            self.world_model_train_state,
            self.world_model_config,
        ) = self.build_world_model(wm_rng)

        self.tokenizer, self.tokenizer_train_state = self.build_tokenizer()

        self.buffer, self.buffer_state, self.buffer_add = self.build_buffer()

        self.wm_loss_fn = self.build_wm_loss_fn()

        self.wm_rollout = self.build_wm_rollout(self.cfg.wm_config.decode_strategy)
        self.wm_rollout_compare = self.build_wm_rollout(decode_strategy="sinkhorn")

        self.wm_rollout_with_actions = self.build_wm_rollout_with_actions(
            self.cfg.wm_config.decode_strategy
        )
        self.wm_rollout_with_actions_compare = self.build_wm_rollout_with_actions(
            decode_strategy="sinkhorn"
        )

        self.batch_hungarian = jax.vmap(Trainer.single_hungarian)
        self.batch_sinkhorn = jax.vmap(
            Trainer.single_sinkhorn, in_axes=(0, None, None, None)
        )

    def build_environment(self, batch_size: int):
        cfg = self.cfg

        if cfg.env_config.env_type == "craftax":
            from craftax.craftax_classic.constants import Achievement, Action

            env = craftax_env.make_craftax_env_from_name(
                "Craftax-Classic-Pixels-v1", auto_reset=True
            )
            env_params = env.default_params
            num_actions = env.action_space(env_params).n

            env = LogWrapper(env)
            env = AutoResetEnvWrapper(env)
            env = BatchEnvWrapper(env, batch_size)

        elif cfg.env_config.env_type == "craftax_full":
            from craftax.craftax.constants import Achievement, Action

            env = craftax_env.make_craftax_env_from_name(
                "Craftax-Pixels-v1", auto_reset=True
            )
            env_params = env.default_params
            num_actions = env.action_space(env_params).n

            env = LogWrapper(env)
            env = AutoResetEnvWrapper(env)
            env = BatchEnvWrapper(env, batch_size)

        else:
            raise ValueError(
                f"Environment type {cfg.env_config.env_type} not supported."
            )

        return env, env_params, num_actions, Achievement, Action

    def build_rollout(self, env, env_params):
        def python_scan(func, out_axes=None, length=0):
            def return_func(carry, xs):
                ys = []
                for i in range(length):
                    carry, y = func(carry, xs[i])
                    ys.append(y)
                return carry, jax.tree.map(lambda *x: jnp.stack(x, axis=1), *ys)

            return return_func

        cond_deco = (
            functools.partial(nnx.jit, static_argnames=("horizon",))
            if env_params is not None
            else (lambda x: x)
        )
        cond_scan = nnx.scan if env_params is not None else python_scan

        @cond_deco
        def rollout(
            agent,
            agent_state,
            curr_obs,
            curr_done,
            env_state,
            horizon,
            rollout_rng,
        ):
            def one_step(state, rng):
                obs, done, env_state, agent_state = state

                pi, value, agent_state = jax.lax.stop_gradient(
                    agent(
                        obs[:, None, ...],
                        done[:, None, ...],
                        agent_state,
                    )
                )
                rng, sample_rng = jax.random.split(rng)
                action, log_prob = pi.sample_and_log_prob(seed=sample_rng)

                action = action.squeeze(axis=1)
                log_prob = log_prob.squeeze(axis=1)
                value = value.squeeze(axis=1)

                rng, step_rng = jax.random.split(rng)
                next_obs, env_state, reward, done, info = env.step(
                    step_rng, env_state, action, env_params
                )
                return (next_obs, done, env_state, agent_state), (
                    obs,
                    action,
                    log_prob,
                    value,
                    reward,
                    done,
                    info,
                )

            (curr_obs, curr_done, env_state, agent_state), (
                obs,
                action,
                log_prob,
                value,
                reward,
                done,
                info,
            ) = cond_scan(
                one_step, out_axes=(nnx.transforms.iteration.Carry, 1), length=horizon
            )(
                (
                    curr_obs,
                    curr_done,
                    env_state,
                    agent_state,
                ),
                jax.random.split(rollout_rng, horizon),
            )
            _, last_value, _ = jax.lax.stop_gradient(
                agent(
                    curr_obs[:, None, ...],
                    curr_done[:, None, ...],
                    agent_state,
                )
            )
            return (curr_obs, curr_done, env_state, agent_state), (
                obs,
                action,
                log_prob,
                jnp.concatenate((value, last_value), axis=1),
                reward,
                done,
                info,
            )

        return rollout

    def build_agent(self):
        cfg = self.cfg

        agent = Agent(
            num_actions=self.num_actions,
            resize=cfg.env_config.env_type != "craftax",
            ac_params=cfg.ac_config.params,
            rngs=nnx.Rngs(cfg.seed),
        )

        tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(learning_rate=cfg.ac_config.learning_rate, eps=1e-5),
        )
        policy_train_state = nnx.Optimizer(agent, tx)

        return agent, policy_train_state

    def build_tokenizer(self):
        cfg = self.cfg

        if cfg.token_config.tokenizer_type == "nnt":
            tokenizer = NearestNeighborTokenizer(
                cfg.token_config.params.codebook_size,
                cfg.token_config.params.patch_size,
                cfg.token_config.params.grid_row,
                cfg.token_config.params.grid_col,
                cfg.token_config.params.threshold,
            )
            return tokenizer, None
        elif cfg.token_config.tokenizer_type == "nnt_superpatch":
            tokenizer = SuperpatchTokenizer(
                cfg.token_config.params.codebook_size,
                cfg.token_config.params.patch_size,
                cfg.token_config.params.threshold,
                cfg.token_config.params.superpatch_height,
                cfg.token_config.params.superpatch_width,
            )
            return tokenizer, None
        else:
            raise ValueError(
                f"Tokenizer type {cfg.token_config.tokenizer_type} not supported."
            )

    def build_world_model(self, rng):
        cfg = self.cfg

        config = GPT2WorldModelConfig(
            num_actions=self.num_actions,
            **asdict(cfg.wm_config.params),
        )
        input_shape = (cfg.batch_size, config.max_tokens)
        world_model = FlaxGPT2WorldModelModule(
            config,
            reward_loss_coef=cfg.wm_config.reward_loss_coef,
            termination_loss_coef=cfg.wm_config.termination_loss_coef,
        )
        rng, init_weights_rng = jax.random.split(rng)
        world_model_params = world_model.init_weights(init_weights_rng, input_shape)

        world_model_tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(cfg.wm_config.learning_rate, eps=1e-5),
        )
        world_model_train_state = TrainState.create(
            apply_fn=world_model.apply,
            params=world_model_params,
            tx=world_model_tx,
        )
        return world_model, world_model_params, world_model_train_state, config

    def build_buffer(self):
        cfg = self.cfg

        buffer = fbx.make_trajectory_buffer(
            add_batch_size=cfg.batch_size,
            sample_batch_size=cfg.batch_size,
            sample_sequence_length=max(cfg.wm_rollout_horizon + 1, cfg.burn_in_horizon),
            period=1,
            min_length_time_axis=max(cfg.wm_rollout_horizon + 1, cfg.burn_in_horizon),
            max_size=cfg.replay_buffer_size,
        )

        buffer_state = buffer.init(
            {
                "obs": jnp.zeros(
                    (
                        cfg.token_config.params.grid_row
                        * cfg.token_config.params.patch_size,
                        cfg.token_config.params.grid_col
                        * cfg.token_config.params.patch_size,
                        3,
                    ),
                    dtype=jnp.float16,
                ),
                "action": jnp.zeros((), dtype=jnp.uint8),
                "reward": jnp.zeros((), dtype=jnp.float16),
                "done": jnp.zeros((), dtype=jnp.bool),
            }
        )

        if self.multi_gpu:
            buffer_state = jax.device_put(
                buffer_state, device=jax.devices()[1], donate=True
            )

        buffer_add = jax.jit(buffer.add, donate_argnums=(0,))

        return buffer, buffer_state, buffer_add

    def build_wm_loss_fn(self):
        @functools.partial(jax.jit, static_argnums=(1,))
        def wm_loss_fn(
            params,
            world_model,
            dropout_key,
            state_action_ids,
            rewards,
            terminations,
        ):
            return world_model.loss(
                params, dropout_key, state_action_ids, rewards, terminations
            )

        return wm_loss_fn

    @staticmethod
    def get_distance_costs(
        grid_row,
        grid_col,
        inventory_height,
        action,
        use_action_distance,
        use_zero_neighbor_distance,
    ):
        H = grid_row
        W = grid_col
        x = jnp.arange(H * W) % W
        y = jnp.arange(H * W) // W

        mask_center = (x > 0) * (x < W - 1) * (y > 0) * (y < H - 1 - inventory_height)

        if use_action_distance:
            action = action[:, None]
            x = x[None, :]
            y = y[None, :]

            distance_costs = (x[:, :, None] - x[:, None, :]) ** 2 + (
                y[:, :, None] - y[:, None, :]
            ) ** 2

            left_mask = x == 0
            right_mask = x == W - 1
            up_mask = y == 0
            down_mask = y == H - 3

            # left_wildcards = ((action == Action.LEFT.value) * left_mask)[:, None, :]
            # right_wildcards = ((action == Action.RIGHT.value) * right_mask)[:, None, :]
            # up_wildcards = ((action == Action.UP.value) * up_mask)[:, None, :]
            # down_wildcards = ((action == Action.DOWN.value) * down_mask)[:, None, :]

            distance_costs = jnp.where(left_wildcards, 100, distance_costs)
            distance_costs = jnp.where(right_wildcards, 100, distance_costs)
            distance_costs = jnp.where(up_wildcards, 100, distance_costs)
            distance_costs = jnp.where(down_wildcards, 100, distance_costs)
        else:
            distance_costs = (x[:, None] - x[None, :]) ** 2 + (
                y[:, None] - y[None, :]
            ) ** 2

        distance_limit = 4.1
        distance_inf = 100
        distance_costs = jnp.where(
            distance_costs > distance_limit, distance_inf, distance_costs
        )

        if use_zero_neighbor_distance:
            distance_costs = jnp.where(distance_costs <= 1.1, 0, distance_costs)

        return distance_costs, mask_center

    @staticmethod
    def single_hungarian(cost_matrix):
        geom = Geometry(
            cost_matrix=cost_matrix,
        )
        cost, hungarian_output = hungarian(geom)
        result = hungarian_output.matrix.transpose().sort_indices().indices[:, 1]
        return result, cost, hungarian_output.matrix

    @staticmethod
    def greedy_single_step(i, output_matrix):
        target = output_matrix.argmax(axis=-2, keepdims=True)
        plan = jnp.zeros_like(output_matrix)
        plan = jnp.put_along_axis(plan, target, 1, axis=-2, inplace=False)

        output_conflict = jnp.where(plan, output_matrix, -jnp.inf)

        source = output_conflict.argmax(axis=-1, keepdims=True)
        plan_win = jnp.zeros_like(output_conflict)
        plan_win = jnp.put_along_axis(plan_win, source, 1, axis=-1, inplace=False)
        plan_win = plan * plan_win
        plan_lose = plan * (1 - plan_win)
        output_matrix = jnp.where(plan_lose, -jnp.inf, output_matrix)
        return output_matrix

    @staticmethod
    def greedy_argmax(output_matrix, tokens_per_block):
        output_matrix = jax.lax.fori_loop(
            0, tokens_per_block, Trainer.greedy_single_step, output_matrix
        )
        return output_matrix.argmax(axis=0)

    @staticmethod
    def single_sinkhorn(cost_matrix, L, sinkhorn_epsilon, tokens_per_block):
        geom = Geometry(
            cost_matrix=cost_matrix,
            epsilon=sinkhorn_epsilon,
            relative_epsilon="std",
        )
        output = linear.solve(geom)
        return (
            Trainer.greedy_argmax(output.matrix.at[:, L:].set(0), tokens_per_block),
            output.matrix,
        )

    def solve_optimal_transport(
        self,
        input_ids,
        tokens_per_block,
        next_state_logits,
        next_state_ids,
        decode_strategy="sinkhorn",
        output_debug=False,
    ):
        curr_state_ids = input_ids[:, -tokens_per_block:-1]

        action = input_ids[:, -1]
        distance_costs, mask_center = Trainer.get_distance_costs(
            self.cfg.token_config.params.grid_row,
            self.cfg.token_config.params.grid_col,
            2 if self.cfg.env_config.env_type == "craftax" else 4,
            action,
            self.cfg.wm_config.use_action_distance,
            self.cfg.wm_config.use_zero_neighbor_distance,
        )

        costs = (
            distance_costs * self.cfg.wm_config.distance_coef
            + jnp.take_along_axis(
                -jax.nn.softmax(next_state_logits[..., None, :, :]),
                curr_state_ids[..., :, None, None],
                axis=-1,
            ).squeeze(axis=-1)
            - self.cfg.wm_config.trash_cost
        )  # (*B, L, L)

        pred_costs = jnp.take_along_axis(
            -jax.nn.softmax(next_state_logits[..., :, :]),  # (*B, L, C)
            next_state_ids[..., :, None],  # (*B, L, 1)
            axis=-1,
        ).squeeze(axis=-1)[
            ..., None, :
        ]  # (*B, 1, L)
        pred_costs = jnp.repeat(
            pred_costs, self.cfg.wm_config.num_dummy, axis=-2
        )  # (*B, D, L)

        shape = pred_costs.shape

        eye = jnp.eye(shape[-1])

        pred_costs = (1 - eye) * 100 + eye * pred_costs

        trash_costs = jnp.zeros((*shape[:-2], shape[-1], shape[-2]))  # (*B, L, D)
        dummy_costs = jnp.zeros((*shape[:-2], shape[-2], shape[-2]))  # (*B, D, D)

        final_costs = jnp.concatenate(
            (
                jnp.concatenate((costs, trash_costs), axis=-1),
                jnp.concatenate((pred_costs, dummy_costs), axis=-1),
            ),
            axis=-2,
        )  # (*B, L + D, L + D)

        if decode_strategy == "hungarian":
            next_state_from, hungarian_costs, hungarian_output = self.batch_hungarian(
                final_costs
            )
            partial_transport = None
        elif decode_strategy == "sinkhorn":
            next_state_from, partial_transport = self.batch_sinkhorn(
                final_costs,
                shape[-1],
                self.cfg.wm_config.sinkhorn_epsilon,
                tokens_per_block,
            )
            hungarian_costs = None
            hungarian_output = None

        next_state_from = next_state_from[..., : tokens_per_block - 1]

        if output_debug:
            return (
                next_state_from,
                mask_center,
                distance_costs,
                costs,
                pred_costs,
                final_costs,
                partial_transport,
                hungarian_costs,
                hungarian_output,
            )
        else:
            return next_state_from, mask_center

    def build_wm_rollout(self, decode_strategy: str = "original"):
        @functools.partial(
            nnx.jit,
            static_argnames=(
                "world_model",
                "tokenizer",
                "horizon",
                "max_tokens",
                "tokens_per_block",
            ),
        )
        def wm_rollout(
            agent,
            agent_state,
            curr_obs,
            curr_done,
            world_model,
            world_model_params,
            tokenizer,
            codebook,
            codebook_size,
            horizon,
            max_tokens,
            tokens_per_block,
            rollout_rng,
        ):
            def one_step(state, rng):
                obs, done, agent_state, position_ids, past_key_values = state

                pi, value, agent_state = jax.lax.stop_gradient(
                    agent(obs[:, None, ...], done[:, None, ...], agent_state)
                )
                rng, sample_rng = jax.random.split(rng)
                action, log_prob = pi.sample_and_log_prob(seed=sample_rng)

                action = action.squeeze(axis=1)
                log_prob = log_prob.squeeze(axis=1)
                value = value.squeeze(axis=1)

                def imagine_state(
                    rng,
                    world_model,
                    params,
                    input_ids,
                    position_ids,
                    past_key_values,
                ):
                    outputs = world_model(
                        params,
                        input_ids,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                    )

                    state_rng, reward_rng, done_rng = jax.random.split(rng, 3)

                    reward_logits = outputs.reward_logits[:, -1]
                    reward = jax.random.categorical(reward_rng, reward_logits)

                    done_logits = outputs.termination_logits[:, -1]
                    done = jax.random.categorical(done_rng, done_logits)

                    tokens_per_state = tokens_per_block - 1
                    next_state_logits = outputs.observation_logits[
                        :, -tokens_per_state:
                    ]
                    mask_codebook = (
                        jnp.arange(self.cfg.token_config.params.codebook_size)
                        >= codebook_size
                    ) * (-jnp.inf)
                    next_state_logits = next_state_logits + mask_codebook
                    next_state_logits = next_state_logits + jax.random.gumbel(
                        state_rng, next_state_logits.shape
                    )
                    next_state_ids = jnp.argmax(next_state_logits, axis=-1)

                    if decode_strategy == "hungarian" or decode_strategy == "sinkhorn":
                        next_state_from, mask_center = self.solve_optimal_transport(
                            input_ids,
                            tokens_per_block,
                            next_state_logits,
                            next_state_ids,
                            decode_strategy,
                        )

                        curr_state_ids = input_ids[:, -tokens_per_block:-1]
                        should_skip = ~done[:, None]
                        next_state_ids = jnp.where(
                            (next_state_from < tokens_per_block - 1)
                            * mask_center
                            * should_skip,
                            jnp.take_along_axis(
                                curr_state_ids, next_state_from, axis=-1
                            ),
                            next_state_ids,
                        )

                    if self.cfg.wm_config.params.use_spatio_temporal:
                        next_position_ids = position_ids + 2
                    else:
                        next_position_ids = position_ids + tokens_per_block

                    return (
                        next_state_ids,
                        reward,
                        done,
                        next_position_ids,
                        outputs.past_key_values,
                    )

                state_ids = tokenizer(obs, codebook)

                state_action_ids = jnp.concatenate(
                    (state_ids, action[:, None]), axis=-1
                )

                rng, step_rng = jax.random.split(rng)
                (
                    next_state_ids,
                    reward,
                    done,
                    position_ids,
                    past_key_values,
                ) = imagine_state(
                    step_rng,
                    world_model,
                    world_model_params,
                    state_action_ids,
                    position_ids,
                    past_key_values,
                )

                next_obs = tokenizer.decode(next_state_ids, codebook)

                return (
                    next_obs,
                    done,
                    agent_state,
                    position_ids,
                    past_key_values,
                ), (
                    obs,
                    action,
                    log_prob,
                    value,
                    reward,
                    done,
                )

            batch_size = curr_obs.shape[0]
            position_ids = get_default_position_ids(
                batch_size,
                tokens_per_block,
                tokens_per_block,
                self.cfg.wm_config.params.use_spatio_temporal,
            )
            past_key_values = world_model.init_cache(batch_size, max_tokens)
            (curr_obs, curr_done, agent_state, _, _), (
                obs,
                action,
                log_prob,
                value,
                reward,
                done,
            ) = nnx.scan(one_step, out_axes=(nnx.transforms.iteration.Carry, 1))(
                (curr_obs, curr_done, agent_state, position_ids, past_key_values),
                jax.random.split(rollout_rng, horizon),
            )

            _, last_value, _ = jax.lax.stop_gradient(
                agent(curr_obs[:, None, ...], curr_done[:, None, ...], agent_state)
            )

            return (
                obs,
                action,
                log_prob,
                jnp.concatenate((value, last_value), axis=1),
                reward,
                done,
            )

        return wm_rollout

    def build_wm_rollout_with_actions(self, decode_strategy: str = "original"):
        @functools.partial(
            nnx.jit,
            static_argnames=(
                "world_model",
                "tokenizer",
                "horizon",
                "max_tokens",
                "tokens_per_block",
            ),
        )
        def wm_rollout(
            curr_obs,
            curr_done,
            actions,
            world_model,
            world_model_params,
            tokenizer,
            codebook,
            codebook_size,
            horizon,
            max_tokens,
            tokens_per_block,
            rollout_rng,
        ):
            def one_step(state, action, rng):
                obs, done, position_ids, past_key_values = state

                def imagine_state(
                    rng,
                    world_model,
                    params,
                    input_ids,
                    position_ids,
                    past_key_values,
                ):
                    outputs = world_model(
                        params,
                        input_ids,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                    )

                    state_rng, reward_rng, done_rng = jax.random.split(rng, 3)

                    reward_logits = outputs.reward_logits[:, -1]
                    reward = jax.random.categorical(reward_rng, reward_logits)

                    done_logits = outputs.termination_logits[:, -1]
                    done = jax.random.categorical(done_rng, done_logits)

                    tokens_per_state = tokens_per_block - 1
                    next_state_logits = outputs.observation_logits[
                        :, -tokens_per_state:
                    ]
                    mask_codebook = (
                        jnp.arange(self.cfg.token_config.params.codebook_size)
                        >= codebook_size
                    ) * (-jnp.inf)
                    next_state_logits = next_state_logits + mask_codebook
                    next_state_logits = next_state_logits + jax.random.gumbel(
                        state_rng, next_state_logits.shape
                    )
                    next_state_ids = jnp.argmax(next_state_logits, axis=-1)

                    next_state_from = None

                    if decode_strategy == "hungarian" or decode_strategy == "sinkhorn":
                        next_state_from, mask_center = self.solve_optimal_transport(
                            input_ids,
                            tokens_per_block,
                            next_state_logits,
                            next_state_ids,
                            decode_strategy,
                        )

                        curr_state_ids = input_ids[:, -tokens_per_block:-1]
                        should_skip = ~done[:, None]
                        next_state_ids = jnp.where(
                            (next_state_from < tokens_per_block - 1)
                            * mask_center
                            * should_skip,
                            jnp.take_along_axis(
                                curr_state_ids, next_state_from, axis=-1
                            ),
                            next_state_ids,
                        )

                    if self.cfg.wm_config.params.use_spatio_temporal:
                        next_position_ids = position_ids + 2
                    else:
                        next_position_ids = position_ids + tokens_per_block

                    return (
                        next_state_ids,
                        reward,
                        done,
                        next_position_ids,
                        outputs.past_key_values,
                        next_state_from,
                    )

                state_ids = tokenizer(obs, codebook)

                state_action_ids = jnp.concatenate(
                    (state_ids, action[:, None]), axis=-1
                )

                rng, step_rng = jax.random.split(rng)
                (
                    next_state_ids,
                    reward,
                    done,
                    position_ids,
                    past_key_values,
                    next_state_from,
                ) = imagine_state(
                    step_rng,
                    world_model,
                    world_model_params,
                    state_action_ids,
                    position_ids,
                    past_key_values,
                )

                next_obs = tokenizer.decode(next_state_ids, codebook)

                return (
                    next_obs,
                    done,
                    position_ids,
                    past_key_values,
                ), (
                    obs,
                    action,
                    reward,
                    done,
                    next_state_from,
                )

            batch_size = curr_obs.shape[0]
            position_ids = get_default_position_ids(
                batch_size,
                tokens_per_block,
                tokens_per_block,
                self.cfg.wm_config.params.use_spatio_temporal,
            )
            past_key_values = world_model.init_cache(batch_size, max_tokens)
            (curr_obs, curr_done, _, _), (
                obs,
                action,
                reward,
                done,
                next_state_from,
            ) = nnx.scan(
                one_step,
                in_axes=(nnx.transforms.iteration.Carry, 1, 0),
                out_axes=(nnx.transforms.iteration.Carry, 1),
            )(
                (curr_obs, curr_done, position_ids, past_key_values),
                actions,
                jax.random.split(rollout_rng, horizon),
            )

            return (
                obs,
                action,
                reward,
                done,
                next_state_from,
            )

        return wm_rollout

    def get_action_ratio_logs(self, data, prefix):
        _obs, _reset, action, _log_prob, _adv, _tgt = data
        action = action.flatten().astype(jnp.uint8)
        counts = jnp.bincount(action, length=self.num_actions)
        ratios = counts.astype(jnp.float32) / jnp.sum(counts)

        logs = {}
        for i, ratio in enumerate(ratios):
            logs[f"{prefix}/action_{i}"] = ratio

        return logs

    def train(self):
        cfg = self.cfg

        # Reset environment
        rng, env_rng = jax.random.split(self.rng)

        start_time = time.time()
        self.curr_obs, self.env_state = self.env.reset(env_rng, self.env_params)
        end_time = time.time()
        print(f"Reset time: {end_time - start_time:.2f}s")
        self.curr_done = jnp.ones((cfg.batch_size,), dtype=jnp.bool)

        # Reset agent state
        self.agent_state = self.agent.rnn.initialize_carry(cfg.batch_size)

        # Reset tokenizer
        self.codebook = self.tokenizer.init_codebook()
        self.codebook_size = jnp.array(0)

        self.success_rates = jnp.zeros((len(self.Achievement),))
        self.num_episodes = jnp.array(0)

        self.tgt_mean = 0
        self.tgt_std = 0
        self.debiasing = 0

        if cfg.restore_ckpt_path is not None:
            self.restore_state()

        # Start training loop
        step_size = cfg.batch_size * cfg.rollout_horizon
        for step in tqdm(
            range(step_size, cfg.total_env_interactions, step_size),
            desc="Training",
        ):
            # 1. Collect data from environment
            rng, rollout_rng = jax.random.split(rng)
            data, next_agent_state = self.collect_from_env(rollout_rng, step)

            # 2. Update policy on environment data
            policy_env_logs = self.learn_policy(
                data,
                self.agent_state,
                cfg.ac_config.num_real_epochs,
                cfg.ac_config.num_real_minibatches,
            )

            if cfg.wandb_config.enable and len(policy_env_logs) > 0:
                logs = {}
                for k in policy_env_logs[0].keys():
                    logs[f"policy_env/{k}"] = jnp.array(
                        [l[k] for l in policy_env_logs]
                    ).mean()

                logs.update(self.get_action_ratio_logs(data, "policy_env"))

                wandb.log(logs, step=step)

            self.agent_state = next_agent_state

            # 3. Update world model
            rng, sample_rng = jax.random.split(rng)
            self.learn_world_model(sample_rng, step)

            # 4. Update policy on imagined data
            if step >= cfg.warmup_interactions:
                policy_img_logs = []
                world_model_img_logs = []
                for _ in tqdm(
                    range(cfg.ac_config.num_imagination_updates), desc="Imagination"
                ):
                    rng, collect_rng = jax.random.split(rng)
                    (
                        data,
                        imagination_agent_state,
                        single_world_model_log,
                    ) = self.collect_from_wm(collect_rng)

                    single_policy_log = self.learn_policy(
                        data,
                        imagination_agent_state,
                        cfg.ac_config.num_imagination_epochs,
                        1,
                        enable_tqdm=False,
                    )
                    world_model_img_logs.append(single_world_model_log)
                    policy_img_logs.extend(single_policy_log)

                if cfg.wandb_config.enable and len(policy_img_logs) > 0:
                    logs = {}
                    for k in policy_img_logs[0].keys():
                        logs[f"policy_img/{k}"] = jnp.array(
                            [l[k] for l in policy_img_logs]
                        ).mean()
                    for k in world_model_img_logs[0].keys():
                        logs[f"imagine/{k}"] = jnp.array(
                            [l[k] for l in world_model_img_logs]
                        ).mean()

                    logs.update(self.get_action_ratio_logs(data, "imagine_last"))

                    wandb.log(logs, step=step)

                    if (
                        cfg.save_imagination_interval is not None
                        and step % cfg.save_imagination_interval == 0
                    ):
                        wandb.log(
                            {
                                "imagination": wandb.Image(
                                    concat_to_single_image(data[0])
                                ),
                            },
                            step=step,
                        )

            if cfg.wandb_config.enable:
                wandb.log(
                    {
                        "codebook_size": self.codebook_size,
                    },
                    step=step,
                )

                if (
                    cfg.save_codebook_interval is not None
                    and step % cfg.save_codebook_interval == 0
                ):
                    max_image_length = min(self.codebook_size, 4096)
                    wandb.log(
                        {
                            "codebook": wandb.Image(
                                concat_to_single_image(
                                    self.codebook[:max_image_length], sep=1
                                )
                            ),
                        },
                        step=step,
                    )

            if cfg.ckpt_path is not None:
                is_ckpt_step = any(
                    step <= c and step + step_size > c for c in cfg.ckpt_steps
                )
                if is_ckpt_step:
                    with ocp.CheckpointManager(cfg.ckpt_path) as ckpt_mngr:
                        ckpt_mngr.save(
                            step,
                            args=ocp.args.Composite(
                                codebook=ocp.args.ArraySave(self.codebook),
                                codebook_size=ocp.args.ArraySave(self.codebook_size),
                                tgt_mean=ocp.args.ArraySave(self.tgt_mean),
                                tgt_std=ocp.args.ArraySave(self.tgt_std),
                                debiasing=ocp.args.ArraySave(self.debiasing),
                                success_rates=ocp.args.ArraySave(self.success_rates),
                                num_episodes=ocp.args.ArraySave(self.num_episodes),
                                buffer_state=ocp.args.StandardSave(self.buffer_state),
                                policy_train_state=ocp.args.StandardSave(
                                    nnx.split(self.policy_train_state)[1]
                                ),
                                world_model_train_state=ocp.args.StandardSave(
                                    self.world_model_train_state
                                ),
                            ),
                        )

    def collect_from_env(self, rollout_rng, step):
        cfg = self.cfg

        (self.curr_obs, next_done, self.env_state, next_agent_state), (
            obs,
            action,
            log_prob,
            value,
            reward,
            done,
            info,
        ) = self.rollout(
            self.agent,
            self.agent_state,
            self.curr_obs,
            self.curr_done,
            self.env_state,
            cfg.rollout_horizon,
            rollout_rng,
        )

        self.buffer_state = self.buffer_add(
            self.buffer_state,
            {
                "obs": obs.astype(jnp.float16),
                "action": action.astype(jnp.uint8),
                "reward": reward.astype(jnp.float16),
                "done": done,
            },
        )

        if cfg.wandb_config.enable:
            wandb.log(
                {
                    "rollout/reward": reward.mean(),
                    "rollout/done": done.mean(),
                    "rollout/log_prob": log_prob.mean(),
                    "rollout/value": value.mean(),
                    "target_mean": self.tgt_mean,
                    "target_std": self.tgt_std,
                    "debiasing": self.debiasing,
                },
                step=step,
            )

        if info["returned_episode"].any():
            avg_episode_returns = jnp.average(
                info["returned_episode_returns"], weights=info["returned_episode"]
            )
            num_episode_ends = info["returned_episode"].sum()

            weights = jnp.broadcast_to(
                info["returned_episode"][..., None],
                info["returned_episode_achievements"].shape,
            )
            avg_episode_achievements = jnp.average(
                info["returned_episode_achievements"],
                weights=weights,
                axis=range(len(info["returned_episode_achievements"].shape))[:-1],
            )
            score = (
                jnp.exp(jnp.mean(jnp.log(1 + avg_episode_achievements * 100.0))) - 1.0
            )
            self.success_rates = (
                self.success_rates * self.num_episodes
                + avg_episode_achievements * num_episode_ends
            ) / (self.num_episodes + num_episode_ends)
            self.num_episodes = self.num_episodes + num_episode_ends
            acc_score = jnp.exp(jnp.mean(jnp.log(1 + self.success_rates * 100.0))) - 1.0

            if cfg.wandb_config.enable:
                wandb.log(
                    {
                        "rollout/return": avg_episode_returns,
                        "rollout/ends": num_episode_ends,
                        **{
                            f"rollout/achievements/{achievement.name}": avg_episode_achievements[
                                achievement.value
                            ]
                            for achievement in self.Achievement
                        },
                        "rollout/single_score": score,
                        "rollout/accumulated_score": acc_score,
                    },
                    step=step,
                )

        reset = jnp.concatenate((self.curr_done[:, None], done[:, :-1]), axis=1)

        value = value * jnp.maximum(
            self.tgt_std / jnp.maximum(self.debiasing, 1e-1), 1e-1
        ) + self.tgt_mean / jnp.maximum(self.debiasing, 1e-2)

        adv, tgt = calc_adv_tgt(
            reward, done, value, cfg.ac_config.gamma, cfg.ac_config.ld
        )

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        self.curr_done = next_done

        return (obs, reset, action, log_prob, adv, tgt), next_agent_state

    def learn_policy(
        self, data, agent_state, n_epochs, n_minibatches, enable_tqdm=True
    ):
        cfg = self.cfg

        obs, reset, action, log_prob, adv, tgt = data

        mini_logs = []

        for epoch in tqdm(range(n_epochs), desc="Policy", disable=not enable_tqdm):
            for i in range(n_minibatches):
                start_idx = i * (cfg.batch_size // n_minibatches)
                end_idx = (i + 1) * (cfg.batch_size // n_minibatches)

                tgt_mini = tgt[start_idx:end_idx]
                self.tgt_mean = (
                    cfg.ac_config.tgt_discount * self.tgt_mean
                    + (1 - cfg.ac_config.tgt_discount) * tgt_mini.mean()
                )
                self.tgt_std = (
                    cfg.ac_config.tgt_discount * self.tgt_std
                    + (1 - cfg.ac_config.tgt_discount) * tgt_mini.std()
                )

                self.debiasing = (
                    cfg.ac_config.tgt_discount * self.debiasing
                    + (1 - cfg.ac_config.tgt_discount) * 1
                )

                tgt_mini = (
                    tgt_mini - self.tgt_mean / jnp.maximum(self.debiasing, 1e-2)
                ) / jnp.maximum(self.tgt_std / jnp.maximum(self.debiasing, 1e-1), 1e-1)

                loss_fn = lambda model: model.loss(
                    obs[start_idx:end_idx],
                    reset[start_idx:end_idx],
                    agent_state[start_idx:end_idx],
                    action[start_idx:end_idx],
                    log_prob[start_idx:end_idx],
                    adv[start_idx:end_idx],
                    tgt_mini,
                )

                (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
                    self.policy_train_state.model
                )

                self.policy_train_state.update(grads=grads)

                mini_logs.append(
                    {
                        "total_loss": loss,
                        **metrics,
                    }
                )

        return mini_logs

    def learn_world_model(self, rng, step):
        cfg = self.cfg

        mini_logs = []

        # Update tokenizer
        for _ in tqdm(range(cfg.token_config.num_updates), desc="Tokenizer"):
            rng, sample_rng = jax.random.split(rng)
            data = self.buffer.sample(self.buffer_state, sample_rng)
            if self.multi_gpu:
                data = jax.device_get(data)

            obs = data.experience["obs"].astype(jnp.float32)
            self.codebook, self.codebook_size = self.tokenizer.update(
                obs, self.codebook, self.codebook_size
            )

        if cfg.wandb_config.enable and len(mini_logs) > 0:
            logs = {}
            for k in mini_logs[0].keys():
                logs[f"tokenizer/{k}"] = jnp.array([l[k] for l in mini_logs]).mean()

            wandb.log(logs, step=step)

        mini_logs = []

        # Update world model
        for _ in tqdm(range(cfg.wm_config.num_updates), desc="World Model"):
            rng, sample_rng = jax.random.split(rng)
            data = self.buffer.sample(self.buffer_state, sample_rng)
            if self.multi_gpu:
                data = jax.device_get(data)

            for i in range(cfg.wm_config.num_minibatches):
                start_idx = i * (cfg.batch_size // cfg.wm_config.num_minibatches)
                end_idx = (i + 1) * (cfg.batch_size // cfg.wm_config.num_minibatches)

                obs = data.experience["obs"][start_idx:end_idx].astype(jnp.float32)
                action = data.experience["action"][start_idx:end_idx].astype(jnp.int32)
                reward = data.experience["reward"][start_idx:end_idx].astype(
                    jnp.float32
                )
                done = data.experience["done"][start_idx:end_idx]

                B, T, *_ = obs.shape

                state_ids = self.tokenizer(obs, self.codebook)

                state_action_ids = jnp.concatenate(
                    (state_ids, action[:, :, None]), axis=-1
                )
                state_action_ids = state_action_ids.reshape(B, -1)

                rng, dropout_rng = jax.random.split(rng)
                (loss, metrics), grads = jax.value_and_grad(
                    self.wm_loss_fn, has_aux=True
                )(
                    self.world_model_train_state.params,
                    self.world_model,
                    dropout_rng,
                    state_action_ids,
                    reward[:, :-1],
                    done[:, :-1].astype(jnp.int32),
                )
                self.world_model_train_state = (
                    self.world_model_train_state.apply_gradients(grads=grads)
                )
                mini_logs.append(
                    {
                        "total_loss": loss,
                        **metrics,
                    }
                )

        if cfg.wandb_config.enable and len(mini_logs) > 0:
            logs = {}
            for k in mini_logs[0].keys():
                logs[f"world_model/{k}"] = jnp.array([l[k] for l in mini_logs]).mean()

            wandb.log(logs, step=step)

    def collect_from_wm(self, rng, wm_rollout=None):
        if wm_rollout is None:
            wm_rollout = self.wm_rollout
        cfg = self.cfg

        rng, sample_rng = jax.random.split(rng)
        data = self.buffer.sample(self.buffer_state, sample_rng)

        obs = data.experience["obs"].astype(jnp.float32)
        done = data.experience["done"]

        if self.multi_gpu:
            obs = jax.device_get(obs[:, : (cfg.burn_in_horizon + 1)])
            done = jax.device_get(done[:, : (cfg.burn_in_horizon + 1)])

        _, _, imagination_agent_state = self.agent(
            obs[:, : cfg.burn_in_horizon],
            done[:, : cfg.burn_in_horizon],
            self.agent.rnn.initialize_carry(cfg.batch_size),
        )

        curr_obs = obs[:, cfg.burn_in_horizon]
        curr_done = done[:, cfg.burn_in_horizon]

        rng, rollout_rng = jax.random.split(rng)
        (
            obs,
            action,
            log_prob,
            value,
            reward,
            done,
        ) = wm_rollout(
            self.agent,
            imagination_agent_state,
            curr_obs,
            curr_done.astype(jnp.int32),
            self.world_model,
            self.world_model_train_state.params,
            self.tokenizer,
            self.codebook,
            self.codebook_size,
            cfg.wm_rollout_horizon,
            self.world_model_config.max_tokens,
            self.world_model_config.tokens_per_block,
            rollout_rng,
        )

        reset = jnp.concatenate((curr_done[:, None], done[:, :-1]), axis=1).astype(
            jnp.bool
        )

        value = value * jnp.maximum(
            self.tgt_std / jnp.maximum(self.debiasing, 1e-1), 1e-1
        ) + self.tgt_mean / jnp.maximum(self.debiasing, 1e-2)

        adv, tgt = calc_adv_tgt(
            reward, done, value, cfg.ac_config.gamma, cfg.ac_config.ld
        )

        single_log = {
            "reward": reward.mean(),
            "done": done.mean(),
            "log_prob": log_prob.mean(),
            "value": value.mean(),
            "return": (reward.sum(axis=-1) + value[:, -1]).mean(),
        }

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return (
            (obs, reset, action, log_prob, adv, tgt),
            imagination_agent_state,
            single_log,
        )

    def restore_state(self):
        cfg = self.cfg

        with ocp.CheckpointManager(cfg.restore_ckpt_path) as ckpt_mngr:
            step = cfg.restore_ckpt_step
            if step is None:
                step = ckpt_mngr.latest_step()

            restored = ckpt_mngr.restore(
                step,
                args=ocp.args.Composite(
                    codebook=ocp.args.ArrayRestore(),
                    codebook_size=ocp.args.ArrayRestore(),
                    tgt_mean=ocp.args.ArrayRestore(),
                    tgt_std=ocp.args.ArrayRestore(),
                    debiasing=ocp.args.ArrayRestore(),
                    success_rates=ocp.args.ArrayRestore(),
                    num_episodes=ocp.args.ArrayRestore(),
                    buffer_state=ocp.args.StandardRestore(),
                    policy_train_state=ocp.args.StandardRestore(),
                    world_model_train_state=ocp.args.StandardRestore(
                        self.world_model_train_state
                    ),
                ),
            )
            self.codebook = restored.codebook
            self.codebook_size = restored.codebook_size
            self.tgt_mean = restored.tgt_mean
            self.tgt_std = restored.tgt_std
            self.debiasing = restored.debiasing
            self.success_rates = restored.success_rates
            self.num_episodes = restored.num_episodes
            self.buffer_state = TrajectoryBufferState(**restored.buffer_state)
            policy_train_state_graphdef, _ = nnx.split(self.policy_train_state)
            self.policy_train_state = nnx.merge(
                policy_train_state_graphdef, restored.policy_train_state
            )
            self.world_model_train_state = restored.world_model_train_state
        self.agent = self.policy_train_state.model
        if self.multi_gpu:
            self.buffer_state = jax.device_put(
                self.buffer_state, device=jax.devices()[1], donate=True
            )

    def collect_from_wm_with_actions(self, data, rng, wm_rollout_with_actions=None):
        if wm_rollout_with_actions is None:
            wm_rollout_with_actions = self.wm_rollout_with_actions

        cfg = self.cfg

        obs = data.experience["obs"].astype(jnp.float32)
        action = data.experience["action"].astype(jnp.int32)
        reward = data.experience["reward"].astype(jnp.float32)
        done = data.experience["done"]

        curr_obs = obs[:, cfg.burn_in_horizon]
        curr_done = done[:, cfg.burn_in_horizon]

        rng, rollout_rng = jax.random.split(rng)
        (
            obs,
            action,
            reward,
            done,
            next_state_from,
        ) = wm_rollout_with_actions(
            curr_obs,
            curr_done.astype(jnp.int32),
            action[:, cfg.burn_in_horizon :],
            self.world_model,
            self.world_model_train_state.params,
            self.tokenizer,
            self.codebook,
            self.codebook_size,
            action.shape[1] - cfg.burn_in_horizon,
            self.world_model_config.max_tokens,
            self.world_model_config.tokens_per_block,
            rollout_rng,
        )

        return (
            obs,
            action,
            reward,
            done,
            next_state_from,
        )

    def train_partially(self):
        cfg = self.cfg

        # Reset environment
        rng, env_rng = jax.random.split(self.rng)

        start_time = time.time()
        self.curr_obs, self.env_state = self.env.reset(env_rng, self.env_params)
        end_time = time.time()
        print(f"Reset time: {end_time - start_time:.2f}s")
        self.curr_done = jnp.ones((cfg.batch_size,), dtype=jnp.bool)

        # Reset agent state
        self.agent_state = self.agent.rnn.initialize_carry(cfg.batch_size)

        # Reset tokenizer
        self.codebook = self.tokenizer.init_codebook()
        self.codebook_size = jnp.array(0)
        self.unary_count = jnp.zeros((self.tokenizer.codebook_size,))
        self.pairwise_count = jnp.zeros(
            (self.tokenizer.codebook_size, self.tokenizer.codebook_size)
        )

        self.tgt_mean = 0
        self.tgt_std = 0
        self.debiasing = 0

        if cfg.restore_ckpt_path is not None:
            self.restore_state()

        # Start training loop
        for step in tqdm(
            range(0, cfg.total_env_interactions, cfg.batch_size * cfg.rollout_horizon),
            desc="Training",
        ):
            # 1. Collect data from environment
            rng, rollout_rng = jax.random.split(rng)
            data, next_agent_state = self.collect_from_env(
                rollout_rng, step + cfg.batch_size * cfg.rollout_horizon
            )

            self.agent_state = next_agent_state

            # 4. Update policy on imagined data
            policy_img_logs = []
            for _ in tqdm(
                range(cfg.ac_config.num_imagination_updates), desc="Imagination"
            ):
                rng, collect_rng = jax.random.split(rng)
                data, imagination_agent_state, _ = self.collect_from_wm(collect_rng)

                (
                    data_transport,
                    imagination_agent_state_transport,
                    _,
                ) = self.collect_from_wm(collect_rng, self.wm_rollout_compare)

                single_log = self.learn_policy(
                    data,
                    imagination_agent_state,
                    cfg.ac_config.num_imagination_epochs,
                    1,
                    enable_tqdm=False,
                )
                policy_img_logs.extend(single_log)

            if cfg.wandb_config.enable and len(policy_img_logs) > 0:
                logs = {}
                for k in policy_img_logs[0].keys():
                    logs[f"policy_img/{k}"] = jnp.array(
                        [l[k] for l in policy_img_logs]
                    ).mean()

                wandb.log(logs, step=step + cfg.batch_size * cfg.rollout_horizon)
                wandb.log(
                    {
                        "imagination": wandb.Image(concat_to_single_image(data[0], 1)),
                        "imagination_transport": wandb.Image(
                            concat_to_single_image(data_transport[0], 1)
                        ),
                    },
                    step=step + cfg.batch_size * cfg.rollout_horizon,
                )

    def imagination_comparison(self):
        cfg = self.cfg

        # Reset environment
        rng, env_rng = jax.random.split(self.rng)

        start_time = time.time()
        self.curr_obs, self.env_state = self.env.reset(env_rng, self.env_params)
        end_time = time.time()
        print(f"Reset time: {end_time - start_time:.2f}s")
        self.curr_done = jnp.ones((cfg.batch_size,), dtype=jnp.bool)

        # Reset agent state
        self.agent_state = self.agent.rnn.initialize_carry(cfg.batch_size)

        # Reset tokenizer
        self.codebook = self.tokenizer.init_codebook()
        self.codebook_size = jnp.array(0)
        self.unary_count = jnp.zeros((self.tokenizer.codebook_size,))
        self.pairwise_count = jnp.zeros(
            (self.tokenizer.codebook_size, self.tokenizer.codebook_size)
        )

        self.tgt_mean = 0
        self.tgt_std = 0
        self.debiasing = 0

        if cfg.restore_ckpt_path is not None:
            self.restore_state()

        # Start training loop
        for step in tqdm(
            range(0, cfg.total_env_interactions, cfg.batch_size * cfg.rollout_horizon),
            desc="Training",
        ):
            # 1. Collect data from environment
            rng, rollout_rng = jax.random.split(rng)
            data, next_agent_state = self.collect_from_env(
                rollout_rng, step + cfg.batch_size * cfg.rollout_horizon
            )

            self.agent_state = next_agent_state

            # 4. Update policy on imagined data
            for _ in tqdm(
                range(cfg.ac_config.num_imagination_updates), desc="Imagination"
            ):
                rng, collect_rng = jax.random.split(rng)

                rng, sample_rng = jax.random.split(rng)
                buffer_data = self.buffer.sample(self.buffer_state, sample_rng)

                data = self.collect_from_wm_with_actions(buffer_data, collect_rng)

                data_transport = self.collect_from_wm_with_actions(
                    buffer_data, collect_rng, self.wm_rollout_with_actions_compare
                )

            if cfg.wandb_config.enable:
                wandb.log(
                    {
                        "original": wandb.Image(
                            concat_to_single_image(
                                buffer_data.experience["obs"].astype(jnp.float32)[
                                    :, cfg.burn_in_horizon :
                                ],
                                1,
                            )
                        ),
                        "imagination": wandb.Image(concat_to_single_image(data[0], 1)),
                        "imagination_transport": wandb.Image(
                            concat_to_single_image(data_transport[0], 1)
                        ),
                    },
                    step=step + cfg.batch_size * cfg.rollout_horizon,
                )

            import ipdb

            ipdb.set_trace()


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

    trainer.train()


if __name__ == "__main__":
    main()
