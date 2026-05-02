from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ActorCriticParams:
    eps: float = 0.2

    td_loss_coef: float = 2.0
    ent_loss_coef: float = 0.01


@dataclass
class ActorCriticConfig:
    params: ActorCriticParams = ActorCriticParams()

    num_real_epochs: int = 4
    num_real_minibatches: int = 8
    num_imagination_updates: int = 150
    num_imagination_epochs: int = 1

    gamma: float = 0.925
    ld: float = 0.625
    tgt_discount: float = 0.95
    ent_loss_coef_real: float = 0.01
    ent_loss_coef_imagination: float = 0.01

    learning_rate: float = 0.00045


@dataclass
class WorldModelParams:
    tokens_per_block: int = 82
    max_blocks: int = 20
    vocab_size: int = 4096
    n_positions: int = 82 * 20
    n_embd: int = 128
    n_layer: int = 3
    n_head: int = 8
    n_inner = None  # defaults to 4 * n_embd
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    use_absolute_embedding: bool = False
    use_spatio_temporal: bool = False


@dataclass
class WorldModelConfig:
    params: WorldModelParams = WorldModelParams()

    num_updates: int = 500
    num_minibatches: int = 3

    reward_loss_coef: float = 1.0
    termination_loss_coef: float = 1.0

    num_dummy: int = 0
    distance_coef: float = 1.0
    trash_cost: float = 1.0
    sinkhorn_epsilon: float = 1e-5
    use_zero_neighbor_distance: bool = False
    use_action_distance: bool = False

    learning_rate: float = 0.001

    decode_strategy: str = "original"


@dataclass
class TokenizerParams:
    codebook_size: int = 4096
    patch_size: int = 7
    patch_channels: int = 3
    grid_row: int = 9
    grid_col: int = 9
    threshold: Optional[float] = 0.75
    superpatch_height: int = 1
    superpatch_width: int = 2


@dataclass
class TokenizerConfig:
    tokenizer_type: str = "nnt"
    params: TokenizerParams = TokenizerParams()

    tokenize_unit: str = "image"

    num_updates: int = 25
    num_minibatches: int = 3

    learning_rate: float = 0.001


@dataclass
class WandBConfig:
    enable: bool = True
    project_name: str = "twm_reproduce"
    exp_name: str = "twm"
    group_name: str = "twm"


@dataclass
class EnvConfig:
    env_type: str = "craftax"
    env_name: str = "Craftax-Classic-Pixels-v1"


@dataclass
class TrainConfig:
    seed: int = 0
    total_env_interactions: int = 1_000_000
    warmup_interactions: int = 200_000

    restore_ckpt_path: Optional[str] = None
    restore_ckpt_step: Optional[int] = None
    ckpt_path: Optional[str] = None
    ckpt_steps: List[int] = field(
        default_factory=lambda: [50000, 200000, 500000, 1000000]
    )

    save_codebook_interval: Optional[int] = 20 * 48 * 96
    save_imagination_interval: Optional[int] = 20 * 48 * 96

    replay_buffer_size: int = 128_000

    batch_size: int = 48
    rollout_horizon: int = 96
    wm_rollout_horizon: int = 20
    max_grad_norm: float = 0.5

    burn_in_horizon: int = 5

    wandb_config: WandBConfig = WandBConfig()
    ac_config: ActorCriticConfig = ActorCriticConfig()
    wm_config: WorldModelConfig = WorldModelConfig()
    token_config: TokenizerConfig = TokenizerConfig()
    env_config: EnvConfig = EnvConfig()

    tqdm_disable: bool = False
    tqdm_interval: float = 0.1
