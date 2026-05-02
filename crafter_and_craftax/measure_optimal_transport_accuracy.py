import itertools

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from ott.geometry.geometry import Geometry
from ott.solvers import linear
from ott.tools.unreg import hungarian
from PIL import Image
import pyrallis
from tqdm import tqdm


from configs import TrainConfig
from nets.nnt import NearestNeighborTokenizer
from train import Trainer
from utils.visualization import concat_to_single_image


# TODO: The codebook entries of the saved model must be manually annotated here.
# Use `visualize_codebook.py` to visualize the codebook entries.
cows = jnp.array(
    [
        1,
        36,
        39,
        71,
        86,
        98,
        102,
        106,
        127,
        128,
        141,
        156,
        188,
        224,
        358,
        382,
        401,
        439,
    ]
)
zombies = jnp.array(
    [33, 37, 43, 58, 60, 81, 118, 121, 129, 155, 189, 201, 252, 377, 415, 419, 439]
)
skeletons = jnp.array(
    [45, 55, 72, 84, 120, 160, 165, 166, 175, 176, 206, 218, 269, 419]
)


def single_hungarian(cost_matrix):
    geom = Geometry(
        cost_matrix=cost_matrix,
    )
    cost, hungarian_output = hungarian(geom)
    result = hungarian_output.matrix.transpose().sort_indices().indices[:, 1]
    return result, cost, hungarian_output.matrix


batch_hungarian = jax.vmap(single_hungarian)


def create_cost_matrix(state_ids, predicted_ids, dx, dy, heading_idx):
    state_ids = state_ids[:, : (7 * 9)]
    predicted_ids = predicted_ids[:, : (7 * 9)]

    # x, y position of 7x9
    x = jnp.arange(7 * 9) % 9
    y = jnp.arange(7 * 9) // 9

    extra_x = jnp.arange(11 * 13) % 13 - 2  # -2, -1, and 9, 10 for extra edges
    extra_y = jnp.arange(11 * 13) // 13 - 2  # -2, -1, and 7, 8 for extra edges

    mask = (extra_x >= 0) & (extra_x < 9) & (extra_y >= 0) & (extra_y < 7)
    extra_x = extra_x[~mask]
    extra_y = extra_y[~mask]

    # add position for outside edges
    x = jnp.concatenate(
        (
            x,
            extra_x,
            jnp.array([100]),  # very far location for dead creature
        )
    )
    y = jnp.concatenate(
        (
            y,
            extra_y,
            jnp.array([100]),  # very far location for dead creature
        )
    )

    extra_len = (~mask).sum()

    batch_size = state_ids.shape[0]
    cost = jnp.zeros((batch_size, 7 * 9 + extra_len + 1, 7 * 9 + extra_len + 1))

    for creatures in [cows, zombies, skeletons]:
        state_ids_creature = jnp.isin(state_ids, creatures)
        predicted_ids_creature = jnp.isin(predicted_ids, creatures)

        state_ids_creature_without_extra = jnp.concatenate(
            (
                state_ids_creature,
                jnp.zeros((batch_size, extra_len + 1), dtype=jnp.bool),
            ),
            axis=1,
        )
        state_ids_creature_with_extra = jnp.concatenate(
            (
                state_ids_creature,
                jnp.ones((batch_size, extra_len + 1), dtype=jnp.bool),
            ),
            axis=1,
        )
        predicted_ids_creature_without_extra = jnp.concatenate(
            (
                predicted_ids_creature,
                jnp.zeros((batch_size, extra_len + 1), dtype=jnp.bool),
            ),
            axis=1,
        )
        predicted_ids_creature_with_extra = jnp.concatenate(
            (
                predicted_ids_creature,
                jnp.ones((batch_size, extra_len + 1), dtype=jnp.bool),
            ),
            axis=1,
        )

        matched = (
            state_ids_creature_with_extra[:, :, None]
            & predicted_ids_creature_with_extra[:, None, :]
        )
        matched = matched & (
            (x[None, :, None] - (x[None, None, :] + dx[:, None, None])) ** 2
            + (y[None, :, None] - (y[None, None, :] + dy[:, None, None])) ** 2
            <= 1
        )  # only allow matching with distance less than 1
        matched = matched.at[jnp.arange(heading_idx.shape[0]), heading_idx, -1].set(
            1
        )  # target_idx creature can die

        cost = cost + jnp.where(
            state_ids_creature_without_extra[:, :, None]
            | predicted_ids_creature_without_extra[:, None, :],
            ~matched,  # if source or target is a creature, then cost is 0 if they match, 1 otherwise
            0,  # if neither source nor target is a creature, then cost is 0
        )

    return cost


def check_prediction_correctness(
    state_ids, action, next_state_ids, predicted_ids, curr_obs, next_obs, codebook, i
):
    # TODO: The codebook entries of the saved model must be manually annotated here.
    # Use `visualize_codebook.py` to visualize the codebook entries.
    creatures = jnp.concatenate([cows, zombies, skeletons])

    empty_inventory = 10
    healths = jnp.array(
        [
            68,  # 1
            34,  # 2
            61,  # 3
            69,  # 4
            53,  # 5
            56,  # 6
            47,  # 7
            67,  # 8
            6,  # 9
        ]
    )
    hungers = jnp.array(
        [
            182,  # 1
            181,  # 2
            109,  # 3
            94,  # 4
            87,  # 5
            48,  # 6
            11,  # 7
            7,  # 8
            19,  # 9
        ]
    )
    thirsts = jnp.array(
        [
            132,  # 1
            103,  # 2
            92,  # 3
            88,  # 4
            54,  # 5
            16,  # 6
            8,  # 7
            20,  # 8
            30,  # 9
        ]
    )
    energies = jnp.array(
        [
            278,  # 1
            254,  # 2
            237,  # 3
            236,  # 4
            140,  # 5
            139,  # 6
            57,  # 7
            22,  # 8
            9,  # 9
        ]
    )
    seeds = jnp.array(
        [
            empty_inventory,
            32,  # 1
            74,  # 2
            248,  # 3
            273,  # 4
            274,  # 5
            275,  # 6
            271,  # 7
            276,  # 8
            277,  # 9
        ]
    )

    grass = jnp.array([2, 25, 83, 99, 108])
    tree = jnp.array([18, 26, 93, 136])
    sand = jnp.array([3, 24, 97, 114, 144])
    path = jnp.array([0, 119])
    stone = jnp.array([12])
    water = jnp.array([5, 96, 105, 112, 115])
    coal = jnp.array([17, 28, 133, 143, 145])
    iron = jnp.array([15, 51, 117, 152])
    diamond = jnp.array([31, 158, 256, 265, 266, 270])
    lava = jnp.array([40, 41, 70, 150, 153, 172, 299])
    plant = jnp.array([29, 95])
    table = jnp.array([50, 131, 177])
    furnace = jnp.array([305, 307, 315, 339])
    noise = jnp.array(
        [
            104,
            138,
            184,
            185,
            186,
            187,
            191,
            192,
            193,
            194,
            195,
            196,
            197,
            198,
            199,
            202,
            203,
            207,
            208,
            209,
            210,
            211,
            212,
            213,
            214,
            216,
            217,
            220,
            221,
            223,
            227,
            235,
            238,
            255,
            257,
            258,
            259,
            264,
            267,
            272,
            283,
            286,
            288,
            290,
            295,
            301,
            302,
            308,
            309,
            310,
            312,
            313,
            314,
            318,
            320,
            321,
            324,
            325,
            327,
            337,
            338,
            340,
            341,
            342,
            343,
            347,
            348,
            351,
            352,
            356,
            357,
            359,
            363,
            364,
            365,
            366,
            368,
            369,
            370,
            371,
            372,
            374,
            376,
            379,
            380,
            386,
            387,
            390,
            393,
            397,
            398,
            400,
            402,
            406,
            407,
            409,
            410,
            411,
            412,
            413,
            416,
            417,
            420,
            421,
            425,
            426,
            431,
            433,
            434,
            435,
            436,
            437,
            438,
        ]
    )

    grass_up_arrow = jnp.array([46, 163, 168])
    grass_down_arrow = jnp.array([59, 91, 167, 243])
    grass_left_arrow = jnp.array([80, 157, 159])
    grass_right_arrow = jnp.array([62, 161, 244])
    sand_up_arrow = jnp.array([169, 322])
    sand_down_arrow = jnp.array([125])
    sand_left_arrow = jnp.array([253])
    sand_right_arrow = jnp.array([64, 180])
    path_up_arrow = jnp.array([75, 204, 229])
    path_down_arrow = jnp.array([78, 222])
    path_left_arrow = jnp.array([183, 230, 239])
    path_right_arrow = jnp.array([77, 245])
    water_up_arrow = jnp.array([170, 228])
    water_down_arrow = jnp.array([240, 246, 262, 268])
    water_left_arrow = jnp.array([73, 241, 250])
    water_right_arrow = jnp.array([63, 65, 162, 350])
    lava_up_arrow = jnp.array([76, 234, 260, 263, 328])
    lava_down_arrow = jnp.array([225, 232, 233, 404])
    lava_left_arrow = jnp.array([249, 304, 383, 385])
    lava_right_arrow = jnp.array([130, 251, 280, 367])
    grass_down_right_arrow = jnp.array([317, 375])
    grass_down_left_arrow = jnp.array([428, 440])
    grass_up_right_arrow = jnp.array([381])
    grass_up_left_arrow = jnp.array([408])
    path_up_left_arrow = jnp.array([319, 361])
    path_up_right_arrow = jnp.array([427])
    path_down_right_arrow = jnp.array([242, 326])
    path_down_left_arrow = jnp.array([394])
    lava_up_right_arrow = jnp.array([405])

    arrows = jnp.concatenate(
        [
            grass_up_arrow,
            grass_down_arrow,
            grass_left_arrow,
            grass_right_arrow,
            sand_up_arrow,
            sand_down_arrow,
            sand_left_arrow,
            sand_right_arrow,
            path_up_arrow,
            path_down_arrow,
            path_left_arrow,
            path_right_arrow,
            water_up_arrow,
            water_down_arrow,
            water_left_arrow,
            water_right_arrow,
            lava_up_arrow,
            lava_down_arrow,
            lava_left_arrow,
            lava_right_arrow,
            grass_down_right_arrow,
            grass_down_left_arrow,
            grass_up_right_arrow,
            grass_up_left_arrow,
            path_up_left_arrow,
            path_up_right_arrow,
            path_down_right_arrow,
            path_down_left_arrow,
            lava_up_right_arrow,
        ]
    )

    player_down_grass = jnp.array([4, 89, 107, 377])
    player_down_sand = jnp.array([44, 135, 148, 215, 382])
    player_down_path = jnp.array([66, 124, 171, 261])
    player_up_grass = jnp.array([13, 100, 111, 116, 415])
    player_up_sand = jnp.array([110, 164, 179])
    player_up_path = jnp.array([123, 174, 231])
    player_left_grass = jnp.array([14, 90, 126])
    player_left_sand = jnp.array([52, 134, 146, 178, 190])
    player_left_path = jnp.array([122, 154, 219, 226])
    player_right_grass = jnp.array([21, 58, 85, 113, 151])
    player_right_sand = jnp.array([23, 147, 205])
    player_right_path = jnp.array([82, 173, 200])
    sleeps = jnp.array([27, 42, 101, 142])

    left_players = jnp.concatenate(
        [
            player_left_grass,
            player_left_sand,
            player_left_path,
        ]
    )
    right_players = jnp.concatenate(
        [
            player_right_grass,
            player_right_sand,
            player_right_path,
        ]
    )
    up_players = jnp.concatenate(
        [
            player_up_grass,
            player_up_sand,
            player_up_path,
        ]
    )
    down_players = jnp.concatenate(
        [
            player_down_grass,
            player_down_sand,
            player_down_path,
        ]
    )

    impassable = jnp.concatenate(
        [
            tree,
            stone,
            water,
            coal,
            iron,
            diamond,
            lava,
            plant,
            table,
            furnace,
            creatures,
        ]
    )
    # 24 is darkened sand used as stone
    # stone = jnp.concatenate([stone, jnp.array([24])])

    light_level_groups = [
        grass,
        tree,
        sand,
        path,
        stone,
        water,
        coal,
        iron,
        diamond,
        lava,
        plant,
        table,
        furnace,
        noise,
        grass_up_arrow,
        grass_down_arrow,
        grass_left_arrow,
        grass_right_arrow,
        sand_up_arrow,
        sand_down_arrow,
        sand_left_arrow,
        sand_right_arrow,
        path_up_arrow,
        path_down_arrow,
        path_left_arrow,
        path_right_arrow,
        water_up_arrow,
        water_down_arrow,
        water_left_arrow,
        water_right_arrow,
        lava_up_arrow,
        lava_down_arrow,
        lava_left_arrow,
        lava_right_arrow,
        grass_down_right_arrow,
        grass_down_left_arrow,
        grass_up_right_arrow,
        grass_up_left_arrow,
        path_up_left_arrow,
        path_up_right_arrow,
        path_down_right_arrow,
        path_down_left_arrow,
        lava_up_right_arrow,
        player_down_grass,
        player_down_sand,
        player_down_path,
        player_up_grass,
        player_up_sand,
        player_up_path,
        player_left_grass,
        player_left_sand,
        player_left_path,
        player_right_grass,
        player_right_sand,
        player_right_path,
        sleeps,
    ]

    sleeping = jnp.isin(state_ids, sleeps).any(axis=1)
    target_idx = (
        (3 * 9 + 4)
        - 1 * (action == 1)
        + 1 * (action == 2)
        - 9 * (action == 3)
        + 9 * (action == 4)
    )

    target_token = jnp.take_along_axis(state_ids, target_idx[:, None], axis=-1).squeeze(
        axis=-1
    )

    blocked = jnp.isin(target_token, impassable) | sleeping

    heading_idx = jnp.where(
        action == 5,
        (
            (3 * 9 + 4)
            - 1 * (jnp.isin(state_ids, left_players).any(axis=-1))
            + 1 * (jnp.isin(state_ids, right_players).any(axis=-1))
            - 9 * (jnp.isin(state_ids, up_players).any(axis=-1))
            + 9 * (jnp.isin(state_ids, down_players).any(axis=-1))
        ),
        1000,  # out of index to make no update
    )

    valid_tokens = predicted_ids == next_state_ids
    for group in light_level_groups:
        valid_tokens |= jnp.isin(predicted_ids, group) & jnp.isin(next_state_ids, group)

    # Ignore one edge if player moved
    action_extra_dim = action[:, None]
    x = jnp.arange(81) % 9
    y = jnp.arange(81) // 9
    unpredictable_area = (
        ((action_extra_dim == 1) & (x == 0))
        | ((action_extra_dim == 2) & (x == 8))
        | ((action_extra_dim == 3) & (y == 0))
        | ((action_extra_dim == 4) & (y == 6))
    ) & ~blocked[:, None]
    valid_tokens |= unpredictable_area

    # Calculate amount of shift based on action
    dx = jnp.where(blocked, 0, -1 * (action == 1) + 1 * (action == 2))
    dy = jnp.where(blocked, 0, -1 * (action == 3) + 1 * (action == 4))

    creature_movement_cost = create_cost_matrix(
        state_ids, predicted_ids, dx, dy, heading_idx
    )

    _, cost, _ = batch_hungarian(creature_movement_cost)

    # Allow creatures tokens because it is handled by cost result
    valid_tokens |= jnp.isin(next_state_ids, creatures) | jnp.isin(
        predicted_ids, creatures
    )

    valid_prediction = valid_tokens.all(axis=1) & (cost == 0)

    return valid_prediction


def test_datapoints(
    trainer,
    tokens_per_block,
    state_ids,
    action,
    predicted_logits,
    predicted_ids,
    next_state_ids,
    curr_obs,
    next_obs,
    codebook,
    i,
    visualize=False,
):
    input_ids = jnp.concatenate((state_ids, action[:, None]), axis=-1)
    (
        next_state_from,
        _mask_center,
        distance_costs,
        costs,
        pred_costs,
        final_costs,
        partial_transport,
        hungarian_costs,
        hungarian_output,
    ) = trainer.solve_optimal_transport(
        input_ids,
        tokens_per_block,
        predicted_logits,
        predicted_ids,
        trainer.cfg.wm_config.decode_strategy,
        output_debug=True,
    )

    next_state_from = jnp.where(
        next_state_from >= tokens_per_block - 1, -1, next_state_from
    )

    final_next_state_ids = jnp.where(
        next_state_from == -1,
        predicted_ids,
        jnp.take_along_axis(state_ids, next_state_from, axis=-1),
    )

    state_ids_match = check_prediction_correctness(
        state_ids,
        action,
        next_state_ids,
        final_next_state_ids,
        curr_obs,
        next_obs,
        codebook,
        i,
    )
    state_ids_match_ratio = (final_next_state_ids == next_state_ids).mean(axis=1)

    transformer_match = check_prediction_correctness(
        state_ids,
        action,
        next_state_ids,
        predicted_ids,
        curr_obs,
        next_obs,
        codebook,
        i,
    )
    transformer_match_ratio = (predicted_ids == next_state_ids).mean(axis=1)

    return (
        state_ids_match,
        state_ids_match_ratio,
        transformer_match,
        transformer_match_ratio,
    )


def test_solver(trainer, dataset, codebook):
    cfg = trainer.cfg
    tokens_per_block = cfg.wm_config.params.tokens_per_block

    state_matches = 0
    state_score = 0
    transformer_matches = 0
    transformer_score = 0
    both_correct = 0
    both_incorrect = 0
    ot_only_correct = 0
    transformer_only_correct = 0
    total_count = 0

    state_ids = dataset["curr_token"]
    action = dataset["action"]
    predicted_logits = dataset["transformer_logits"]
    predicted_ids = dataset["transformer_preds"]
    next_state_ids = dataset["next_token"]
    curr_obs = dataset["curr_obs"]
    next_obs = dataset["next_obs"]

    for i in tqdm(range(0, state_ids.shape[0], cfg.batch_size)):
        batch_state_ids = state_ids[i : i + cfg.batch_size]
        batch_action = action[i : i + cfg.batch_size]
        batch_predicted_logits = predicted_logits[i : i + cfg.batch_size]
        batch_predicted_ids = predicted_ids[i : i + cfg.batch_size]
        batch_next_state_ids = next_state_ids[i : i + cfg.batch_size]
        batch_curr_obs = curr_obs[i : i + cfg.batch_size]
        batch_next_obs = next_obs[i : i + cfg.batch_size]

        creatures = jnp.concatenate([cows, zombies, skeletons])
        contains_creature = (jnp.isin(batch_state_ids, creatures)).any(axis=1)
        # TODO: Comment line below to get overall prediction accuracy
        contains_creature |= 1

        (
            state_ids_match,
            state_ids_match_ratio,
            transformer_match,
            transformer_match_ratio,
        ) = test_datapoints(
            trainer,
            tokens_per_block,
            batch_state_ids,
            batch_action,
            batch_predicted_logits,
            batch_predicted_ids,
            batch_next_state_ids,
            batch_curr_obs,
            batch_next_obs,
            codebook,
            i,
            visualize=True,
        )

        both_correct += (state_ids_match & transformer_match).sum(
            where=contains_creature
        )
        both_incorrect += (~state_ids_match & ~transformer_match).sum(
            where=contains_creature
        )
        ot_only_correct += (state_ids_match & ~transformer_match).sum(
            where=contains_creature
        )
        transformer_only_correct += (~state_ids_match & transformer_match).sum(
            where=contains_creature
        )

        state_matches += state_ids_match.sum(where=contains_creature)
        state_score += state_ids_match_ratio.sum(where=contains_creature)
        transformer_matches += transformer_match.sum(where=contains_creature)
        transformer_score += transformer_match_ratio.sum(where=contains_creature)

        total_count += contains_creature.sum()

    state_score /= total_count
    transformer_score /= total_count

    print(f"Total count: {total_count}")
    print(f"Current model matches: {state_matches}, {state_matches / total_count}")
    print(
        f"Comparison matches: {transformer_matches}, {transformer_matches / total_count}"
    )

    return (
        both_correct,
        both_incorrect,
        ot_only_correct,
        transformer_only_correct,
        state_matches,
        state_score,
    )


def load_dataset(path):
    with ocp.CheckpointManager(path) as ckpt_mngr:
        loaded = ckpt_mngr.restore(
            0,
            args=ocp.args.Composite(
                curr_obs=ocp.args.ArrayRestore(),
                curr_token=ocp.args.ArrayRestore(),
                action=ocp.args.ArrayRestore(),
                next_obs=ocp.args.ArrayRestore(),
                next_token=ocp.args.ArrayRestore(),
                transformer_logits=ocp.args.ArrayRestore(),
                transformer_preds=ocp.args.ArrayRestore(),
            ),
        )
    return loaded


def load_datasets(cfg, path, s, e):
    loaded_concat = {
        "curr_obs": [],
        "curr_token": [],
        "action": [],
        "next_obs": [],
        "next_token": [],
        "transformer_logits": [],
        "transformer_preds": [],
    }
    for i in range(s, e + 1):
        loaded = load_dataset(f"{path}{i}")
        for k in loaded_concat.keys():
            loaded_concat[k].append(loaded[k])

    loaded_concat = {k: jnp.concatenate(v, axis=0) for k, v in loaded_concat.items()}

    if cfg.total_env_interactions < loaded_concat["curr_token"].shape[0]:
        loaded_concat["curr_obs"] = loaded_concat["curr_obs"][
            : cfg.total_env_interactions
        ]
        loaded_concat["curr_token"] = loaded_concat["curr_token"][
            : cfg.total_env_interactions
        ]
        loaded_concat["action"] = loaded_concat["action"][: cfg.total_env_interactions]
        loaded_concat["next_obs"] = loaded_concat["next_obs"][
            : cfg.total_env_interactions
        ]
        loaded_concat["next_token"] = loaded_concat["next_token"][
            : cfg.total_env_interactions
        ]
        loaded_concat["transformer_logits"] = loaded_concat["transformer_logits"][
            : cfg.total_env_interactions
        ]
        loaded_concat["transformer_preds"] = loaded_concat["transformer_preds"][
            : cfg.total_env_interactions
        ]

    return loaded_concat


def load_codebook(path):
    with ocp.CheckpointManager(path) as ckpt_mngr:
        loaded = ckpt_mngr.restore(
            ckpt_mngr.latest_step(),
            args=ocp.args.Composite(
                codebook=ocp.args.ArrayRestore(),
                codebook_size=ocp.args.ArrayRestore(),
            ),
        )
    return loaded["codebook"], loaded["codebook_size"]


@pyrallis.wrap()
def main(cfg: TrainConfig):
    cfg.wm_config.num_dummy = 81

    # TODO: Change dataset directory and dataset range
    dataset = load_datasets(cfg, "dataset_dir_", 0, 9)
    print(f"Datapoints: {dataset['curr_token'].shape[0]}")
    unique_actions, unique_counts = jnp.unique_counts(dataset["action"])
    print(f"Unique actions: {unique_actions}")
    print(f"Unique counts: {unique_counts}")

    # TODO: Change model checkpoint directory
    codebook, codebook_size = load_codebook("model_checkpoint_dir")

    trainer = Trainer(cfg)

    test_solver(trainer, dataset, codebook)


if __name__ == "__main__":
    main()
