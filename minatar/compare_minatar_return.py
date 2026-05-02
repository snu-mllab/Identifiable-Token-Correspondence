import os

import matplotlib.pyplot as plt
import numpy as np


def get_result(result_path, ckpt_steps, num_seeds):
    valuess = []

    for s in range(num_seeds):
        try:
            values = []
            for step in ckpt_steps:
                filename = f"{result_path}/seed{s}/{step}.txt"
                with open(filename, "r") as f:
                    values.append(float(f.read()))
        except FileNotFoundError:
            continue
        valuess.append(values)

    valuess = np.array(valuess)

    return valuess


def draw_subplot(
    ax,
    game_name,
    baseline_result_path,
    baseline_get_result,
    ours_result_path,
    ours_get_result,
    ckpt_steps,
    num_seeds,
    first_subplot,
):
    baseline_result = baseline_get_result(
        baseline_result_path,
        ckpt_steps,
        num_seeds,
    )
    ours_result = ours_get_result(
        ours_result_path,
        ckpt_steps,
        num_seeds,
    )

    baseline_mean = baseline_result.mean(axis=0)
    baseline_stderr = baseline_result.std(axis=0) / np.sqrt(baseline_result.shape[0])

    ours_mean = ours_result.mean(axis=0)
    ours_stderr = ours_result.std(axis=0) / np.sqrt(ours_result.shape[0])

    print(
        f"baseline_nums: {baseline_result.shape[0]}, ours_nums: {ours_result.shape[0]}"
    )

    draw_lines(
        "Dedieu et al." if first_subplot else None,
        "blue",
        ax,
        ckpt_steps,
        baseline_mean,
        baseline_stderr,
    )
    draw_lines(
        "ITC (ours)" if first_subplot else None,
        "red",
        ax,
        ckpt_steps,
        ours_mean,
        ours_stderr,
    )

    ax.vlines(
        x=200000,
        ymin=0,
        ymax=1,
        transform=ax.get_xaxis_transform(),
        color="black",
        linestyle="dashed",
    )

    ax.set_xlabel("# of environment steps")
    if first_subplot:
        ax.set_ylabel("Return")
        # ax.legend()

    ax.set_title(game_name)


ckpt_steps = [
    46080,
    96768,
    147456,
    198144,
    248832,
    299520,
    345600,
    396288,
    446976,
    497664,
    548352,
    599040,
    649728,
    695808,
    746496,
    797184,
    847872,
    898560,
    949248,
    999936,
]


def draw_lines(name, color, ax, steps, mean, std):
    ax.fill_between(
        steps,
        mean - std,
        mean + std,
        color=color,
        alpha=0.1,
    )
    ax.plot(steps, mean, color=color, label=name)


fig, axes = plt.subplots(1, 4, figsize=(15, 2.5))

draw_subplot(
    axes[0],
    "Asterix",
    "",  # TODO: put a path of baseline result for Asterix
    get_result,
    "",  # TODO: put a path of ITC result for Asterix
    get_result,
    ckpt_steps,
    10,
    True,
)

draw_subplot(
    axes[1],
    "Breakout",
    "",  # TODO: put a path of baseline result for Breakout
    get_result,
    "",  # TODO: put a path of ITC result for Breakout
    get_result,
    ckpt_steps,
    10,
    False,
)
draw_subplot(
    axes[2],
    "Freeway",
    "",  # TODO: put a path of baseline result for Freeway
    get_result,
    "",  # TODO: put a path of ITC result for Freeway
    get_result,
    ckpt_steps,
    10,
    False,
)
draw_subplot(
    axes[3],
    "SpaceInvaders",
    "",  # TODO: put a path of baseline result for SpaceInvaders
    get_result,
    "",  # TODO: put a path of ITC result for SpaceInvaders
    get_result,
    ckpt_steps,
    10,
    False,
)

fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=2)
os.makedirs("evaluation_plot", exist_ok=True)
plt.savefig("evaluation_plot/plot.pdf", bbox_inches="tight")
