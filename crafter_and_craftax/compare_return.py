import pandas as pd
import math

import matplotlib.pyplot as plt

from utils.load_wandb import get_mean_std, get_dreamer_result


def draw_lines(name, color, ax, steps, mean, std):
    ax.fill_between(
        steps,
        mean - std,
        mean + std,
        color=color,
        alpha=0.1,
    )
    ax.plot(steps, mean, color=color, label=name)


def draw_ends(name, color, ax, mean, std):
    steps = [900_000, 1_000_000]
    ax.fill_between(
        steps,
        [mean - std, mean - std],
        [mean + std, mean + std],
        color=color,
        alpha=0.1,
    )
    ax.plot(steps, [mean, mean], color=color, label=name)


def draw_dashes(name, color, style, ax, mean):
    steps = [0, 1_000_000]
    ax.hlines(
        mean,
        0,
        1,
        transform=ax.get_yaxis_transform(),
        colors=color,
        linestyles=style,
        label=name,
    )
    # ax.plot(steps, [mean, mean], style, color=color, label=name)


baseline_runs = [
    # TODO: List wandb run ids
]

ours_runs = [
    # TODO: List wandb run ids
]

colors = {
    "human": "black",
    "upperbound": "black",
    "baseline": "blue",
    "ours": "red",
    "dreamer": "green",
    "IRIS": "purple",
    "Delta-IRIS": "orange",
    "DART": "grey",
}

baseline_mean, baseline_std = get_mean_std(baseline_runs)
ours_mean, ours_std = get_mean_std(ours_runs)

steps = baseline_mean.index
baseline_mean = baseline_mean[steps < 1_000_000]
baseline_std = baseline_std[steps < 1_000_000]
steps = steps[steps < 1_000_000]
steps_ours = ours_mean.index

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 2))

draw_dashes("Human expert", colors["human"], ":", ax0, 65.0)

draw_ends(
    "Dreamer (*)",
    colors["dreamer"],
    ax0,
    53.2,
    8.0,
)
draw_ends(
    "IRIS (*)",
    colors["IRIS"],
    ax0,
    25.0,
    3.2,
)
draw_ends(
    "Delta-IRIS (*)",
    colors["Delta-IRIS"],
    ax0,
    35.0,
    3.2,
)
draw_ends(
    "DART (*)",
    colors["DART"],
    ax0,
    12.2 / 22 * 100,
    1.67 / 22 * 100,
)
draw_lines(
    "Dedieu et al.",
    colors["baseline"],
    ax0,
    steps,
    baseline_mean["rollout/return"],
    baseline_std["rollout/return"],
)
draw_lines(
    "ITC (ours)",
    colors["ours"],
    ax0,
    steps_ours,
    ours_mean["rollout/return"],
    ours_std["rollout/return"],
)

# Plot score
draw_dashes(
    "",
    colors["human"],
    ":",
    ax1,
    50.5,
)
draw_lines(
    "",
    colors["baseline"],
    ax1,
    steps,
    baseline_mean["rollout/accumulated_score"],
    baseline_std["rollout/accumulated_score"],
)
draw_lines(
    "",
    colors["ours"],
    ax1,
    steps_ours,
    ours_mean["rollout/accumulated_score"],
    ours_std["rollout/accumulated_score"],
)
draw_ends(
    "",
    colors["dreamer"],
    ax1,
    14.5,
    1.6,
)
draw_ends(
    "",
    colors["IRIS"],
    ax1,
    6.66,
    0.0,
)
draw_ends(
    "",
    colors["Delta-IRIS"],
    ax1,
    9.30,
    0.0,
)

ax0.set_xlabel("# of environment steps")
ax0.set_ylabel("Return (%)")
ax0.grid(True)
ax1.set_xlabel("# of environment steps")
ax1.set_ylabel("Score (%)")
ax1.grid(True)

fig.subplots_adjust(right=0.78, wspace=0.25)  # leave space on the right side

# Add a single legend outside the plots
fig.legend(
    loc="upper left",
    bbox_to_anchor=(0.8, 0.9),  # adjust this as needed
)
# fig.legend(loc="best")

plt.savefig("return.pdf", bbox_inches="tight")
