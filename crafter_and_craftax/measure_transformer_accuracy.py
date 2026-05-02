import itertools

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from ott.geometry.geometry import Geometry
from ott.tools.unreg import hungarian
from PIL import Image
import pyrallis
from tqdm import tqdm

from configs import TrainConfig
from nets.world_model import get_default_position_ids
from train import Trainer
from utils.visualization import concat_to_single_image


def visualize_data(
    predicted_ids,
    final_next_state_ids,
    curr_obs,
    next_obs,
    action,
    next_state_ids,
    focus_token=17,
):
    jnp.set_printoptions(precision=6, linewidth=200, suppress=True)

    obs = jnp.stack((curr_obs, next_obs), axis=1)
    img = concat_to_single_image(obs, 1)
    img = Image.fromarray(img)
    img.save("obs.png")

    for i in range(predicted_ids.shape[0]):
        print(f"--- {i}: action {action[i]} ---")
        predicted_ids_reshaped = predicted_ids[i].reshape(9, 9)
        final_next_state_ids_reshaped = final_next_state_ids[i].reshape(9, 9)
        final_state_correct = (final_next_state_ids[i] == next_state_ids[i]).reshape(
            9, 9
        )

        print(f"predicted_ids:\n{predicted_ids_reshaped}")
        print(f"final_next_state_ids:\n{final_next_state_ids_reshaped}")
        print(f"final_state_correct:\n{final_state_correct}")


def single_hungarian(cost_matrix):
    geom = Geometry(
        cost_matrix=cost_matrix,
    )
    cost, hungarian_output = hungarian(geom)
    result = hungarian_output.matrix.transpose().sort_indices().indices[:, 1]
    return result, cost, hungarian_output.matrix


batch_hungarian = jax.vmap(single_hungarian)


# TODO: The codebook entries of the saved model must be manually annotated here.
# Use `visualize_codebook.py` to visualize the codebook entries.
cows = jnp.array(
    [
        18,
        26,
        38,
        41,
        59,
        97,
        101,
        113,
        122,
        12,
        137,
        138,
        139,
        143,
        172,
        348,
        356,
        386,
        390,
        394,
        404,
    ]
)
zombies = jnp.array(
    [34, 40, 42, 43, 50, 92, 133, 154, 219, 235, 236, 259, 279, 313, 389]
)
skeletons = jnp.array([21, 39, 47, 73, 86, 116, 120, 135, 144, 150, 152, 175, 200, 201])


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


def check_prediction_correctness(state_ids, action, next_state_ids, predicted_ids):
    # TODO: The codebook entries of the saved model must be manually annotated here.
    # Use `visualize_codebook.py` to visualize the codebook entries.
    creatures = jnp.concatenate([cows, zombies, skeletons])

    empty_inventory = 9
    healths = jnp.array(
        [
            70,  # 1
            51,  # 2
            68,  # 3
            48,  # 4
            54,  # 5
            55,  # 6
            53,  # 7
            76,  # 8
            5,  # 9
        ]
    )
    hungers = jnp.array(
        [
            191,  # 1
            180,  # 2
            104,  # 3
            95,  # 4
            80,  # 5
            35,  # 6
            22,  # 7
            6,  # 8
            49,  # 9
        ]
    )
    thirsts = jnp.array(
        [
            119,  # 1
            98,  # 2
            96,  # 3
            79,  # 4
            44,  # 5
            24,  # 6
            14,  # 7
            7,  # 8
            31,  # 9
        ]
    )
    energies = jnp.array(
        [
            276,  # 1
            275,  # 2
            271,  # 3
            241,  # 4
            212,  # 5
            193,  # 6
            32,  # 7
            8,  # 8
            13,  # 9
        ]
    )
    seeds = jnp.array(
        [
            empty_inventory,
            45,  # 1
            69,  # 2
            249,  # 3
            264,  # 4
            265,  # 5
            266,  # 6
            267,  # 7
            268,  # 8
            269,  # 9
        ]
    )

    grass = jnp.array([1, 89, 105, 114])
    tree = jnp.array([3, 11, 94, 117])
    sand = jnp.array([25, 36, 84, 109, 131])
    path = jnp.array([2, 125])
    stone = jnp.array([0, 10, 99])
    water = jnp.array([27, 108, 110, 112, 164])
    coal = jnp.array([20, 57, 88, 118])
    iron = jnp.array([28, 87, 124, 126, 199])
    diamond = jnp.array([64, 65, 82, 90, 102, 162, 211])
    lava = jnp.array([17, 77, 145, 153, 196, 208])
    plant = jnp.array([30, 93])
    ripe_plant = jnp.array([393, 397])
    table = jnp.array([111, 217, 244])
    furnace = jnp.array([253, 258, 332])
    noise = jnp.array(
        [
            157,
            168,
            169,
            178,
            179,
            181,
            182,
            183,
            185,
            186,
            187,
            194,
            198,
            204,
            210,
            215,
            216,
            221,
            225,
            226,
            227,
            228,
            229,
            230,
            231,
            232,
            234,
            245,
            255,
            256,
            257,
            260,
            263,
            270,
            277,
            280,
            283,
            285,
            288,
            289,
            290,
            291,
            293,
            294,
            296,
            298,
            299,
            300,
            301,
            302,
            308,
            309,
            311,
            320,
            336,
            337,
            339,
            340,
            341,
            342,
            343,
            344,
            345,
            346,
            351,
            352,
            353,
            354,
            355,
            357,
            359,
            360,
            361,
            362,
            365,
            366,
            367,
            369,
            371,
            374,
            375,
            376,
            379,
            380,
            383,
            384,
            388,
            391,
            392,
            395,
            396,
            398,
            399,
            400,
            403,
            405,
            406,
            409,
            410,
            414,
            422,
            423,
            424,
            425,
            426,
            427,
            428,
            429,
            430,
            432,
        ]
    )

    grass_up_arrow = jnp.array([66, 67, 197, 250])
    grass_down_arrow = jnp.array([72, 115, 141])
    grass_left_arrow = jnp.array([60, 151, 222])
    grass_right_arrow = jnp.array([61, 136, 155, 156])
    sand_up_arrow = jnp.array([])
    sand_down_arrow = jnp.array([74, 239, 278])
    sand_left_arrow = jnp.array([149, 166])
    sand_right_arrow = jnp.array([46])
    path_up_arrow = jnp.array([107, 214])
    path_down_arrow = jnp.array([78, 146, 242])
    path_left_arrow = jnp.array([129, 243])
    path_right_arrow = jnp.array([63, 202, 262])
    water_up_arrow = jnp.array([238, 240])
    water_down_arrow = jnp.array([195, 220, 402])
    water_left_arrow = jnp.array([148, 174, 213])
    water_right_arrow = jnp.array([58, 192, 237, 254])
    lava_up_arrow = jnp.array([274, 284, 287, 338])
    lava_down_arrow = jnp.array([160, 281, 292, 387])
    lava_left_arrow = jnp.array([233, 272, 273, 295])
    lava_right_arrow = jnp.array([261, 377, 411])
    grass_down_right_arrow = jnp.array([297])
    grass_down_left_arrow = jnp.array([333])
    grass_up_right_arrow = jnp.array([412, 413])
    grass_up_left_arrow = jnp.array([324, 372])
    grass_right_left_arrow = jnp.array([382])
    grass_down_up_arrow = jnp.array([415])
    path_up_left_arrow = jnp.array([286])
    path_down_right_arrow = jnp.array([363])
    path_down_left_arrow = jnp.array([206])
    lava_up_right_arrow = jnp.array([209])
    lava_down_right_arrow = jnp.array([431])

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
            grass_right_left_arrow,
            grass_down_up_arrow,
            path_up_left_arrow,
            path_down_right_arrow,
            path_down_left_arrow,
            lava_up_right_arrow,
            lava_down_right_arrow,
        ]
    )

    player_down_grass = jnp.array([19, 85, 106, 142])
    player_down_sand = jnp.array([56, 130, 147, 159, 205])
    player_down_path = jnp.array([71, 173, 189, 304])
    player_up_grass = jnp.array([23, 91, 121, 123, 313])
    player_up_sand = jnp.array([33, 132, 134, 224])
    player_up_path = jnp.array([171, 176, 207])
    player_left_grass = jnp.array([4, 83, 100, 184])
    player_left_sand = jnp.array([29, 161, 167, 223])
    player_left_path = jnp.array([62, 218, 252])
    player_right_grass = jnp.array([16, 81, 103])
    player_right_sand = jnp.array([52, 140, 158, 203])
    player_right_path = jnp.array([170, 188, 190, 246])
    sleeps = jnp.array([12, 37, 127, 177])

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
            ripe_plant,
            table,
            furnace,
            creatures,
        ]
    )
    # 24 is darkened sand used as stone
    stone = jnp.concatenate([stone, jnp.array([36])])

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
        ripe_plant,
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
        grass_right_left_arrow,
        grass_down_up_arrow,
        path_up_left_arrow,
        path_down_right_arrow,
        path_down_left_arrow,
        lava_up_right_arrow,
        lava_down_right_arrow,
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

    # next_target_token = jnp.take_along_axis(
    #     state_ids, target_idx[:, None], axis=-1
    # ).squeeze(axis=-1)
    # jnp.isin(target_token, creatures) & action == 5

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
    rng,
    visualize=False,
):
    tokens = trainer.tokenizer(curr_obs, trainer.codebook)
    state_action_ids = jnp.concatenate((tokens, action[:, None]), axis=-1)
    position_ids = get_default_position_ids(
        tokens.shape[0], 82, 82, trainer.cfg.wm_config.params.use_spatio_temporal
    )
    past_key_values = trainer.world_model.init_cache(tokens.shape[0], 82 * 20)

    outputs = trainer.world_model(
        trainer.world_model_train_state.params,
        state_action_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
    )

    output_logits = outputs.observation_logits[:, -81:]

    mask_codebook = (jnp.arange(4096) >= trainer.codebook_size) * (-jnp.inf)
    output_logits = output_logits + mask_codebook
    output_logits = output_logits + jax.random.gumbel(rng, output_logits.shape)
    output_state_ids = jnp.argmax(output_logits, axis=-1)

    next_state_tokens = trainer.tokenizer(next_obs, trainer.codebook)

    state_ids_match = check_prediction_correctness(
        state_ids,
        action,
        next_state_tokens,
        output_state_ids,
    )
    state_ids_match_ratio = (output_state_ids == next_state_tokens).mean(axis=1)

    if visualize:
        visualize_data(
            predicted_ids,
            output_state_ids,
            curr_obs,
            next_obs,
            action,
            next_state_ids,
        )

    return (
        state_ids_match,
        state_ids_match_ratio,
    )


def test_solver(trainer, dataset):
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

    rng = jax.random.PRNGKey(0)

    for i in tqdm(range(0, state_ids.shape[0], cfg.batch_size)):
        batch_state_ids = state_ids[i : i + cfg.batch_size]
        batch_action = action[i : i + cfg.batch_size]
        batch_predicted_logits = predicted_logits[i : i + cfg.batch_size]
        batch_predicted_ids = predicted_ids[i : i + cfg.batch_size]
        batch_next_state_ids = next_state_ids[i : i + cfg.batch_size]
        batch_curr_obs = curr_obs[i : i + cfg.batch_size]
        batch_next_obs = next_obs[i : i + cfg.batch_size]

        contains_creature = jnp.ones(batch_state_ids.shape[0])
        # TODO: Uncomment this to get prediction accuracy for cases including creatures
        # creatures = jnp.concatenate([cows, zombies, skeletons])
        # contains_creature = (jnp.isin(batch_state_ids, creatures)).any(axis=1)

        # visualize = ((batch_action >= 1) & (batch_action <= 4)).any()
        visualize = False
        rng, datapoints_rng = jax.random.split(rng)
        (
            state_ids_match,
            state_ids_match_ratio,
            # transformer_match,
            # transformer_match_ratio,
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
            datapoints_rng,
            visualize,
        )

        # both_correct += (state_ids_match & transformer_match).sum()
        # both_incorrect += (~state_ids_match & ~transformer_match).sum()
        # ot_only_correct += (state_ids_match & ~transformer_match).sum()
        # transformer_only_correct += (~state_ids_match & transformer_match).sum()

        state_matches += state_ids_match.sum(where=contains_creature)
        state_score += state_ids_match_ratio.sum(where=contains_creature)
        total_count += contains_creature.sum()
        # transformer_matches += transformer_match.sum()
        # transformer_score += transformer_match_ratio.sum()

    state_score /= total_count
    # transformer_score /= state_ids.shape[0]

    print(f"Total count: {total_count}")
    print(f"Baseline model matches: {state_matches}, {state_matches / total_count}")
    print(f"Baseline model score: {state_score}")
    # print(f"PE model matches: {transformer_matches}")
    # print(f"PE model score: {transformer_score}")
    # print(f"Both correct: {both_correct}")
    # print(f"Both incorrect: {both_incorrect}")
    # print(f"Baseline only correct: {ot_only_correct}")
    # print(f"PE only correct: {transformer_only_correct}")

    return state_matches, state_score


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


@pyrallis.wrap()
def main(cfg: TrainConfig):
    # TODO: Change dataset directory and dataset range
    dataset = load_datasets(cfg, "dataset_dir_", 0, 9)
    print(f"Datapoints: {dataset['curr_token'].shape[0]}")
    unique_actions, unique_counts = jnp.unique_counts(dataset["action"])
    print(f"Unique actions: {unique_actions}")
    print(f"Unique counts: {unique_counts}")

    trainer = Trainer(cfg)
    trainer.restore_state()

    test_solver(trainer, dataset)


if __name__ == "__main__":
    main()
