import pandas as pd
import wandb

api = wandb.Api()


def get_mean_std(
    run_names,
    return_key="rollout/return",
    score_key="rollout/accumulated_score",
    rolling_amount=22,
):
    hist_list = []

    for run_name in run_names:
        single_run = api.run(run_name)

        hist = single_run.history(keys=[return_key, score_key])

        hist[return_key] = hist[return_key] / 22 * 100
        hist[return_key] = hist[return_key].rolling(rolling_amount, min_periods=1).mean()

        hist_list.append(hist)

    concatenated = pd.concat(hist_list)

    mean = concatenated.groupby("_step").mean()
    std = concatenated.groupby("_step").std()

    mean["rollout/return"] = mean[return_key]
    mean["rollout/accumulated_score"] = mean[score_key]

    std["rollout/return"] = std[return_key]
    std["rollout/accumulated_score"] = std[score_key]
    
    return mean, std


def get_dreamer_result(
    run_names,
    ours_steps,
    return_key="rollout/return",
    score_key="rollout/accumulated_score",
    rolling_amount=22,
):
    hist_list = []

    for run_name in run_names:
        single_run = api.run(run_name)

        hist = single_run.history(samples=1_000_000, keys=[return_key, score_key])
        hist["binned_step"] = pd.cut(hist["_step"], [0] + ours_steps)

        processed_hist = {}
        mean = hist.groupby("binned_step").mean()
        processed_hist["_step"] = ours_steps[:-1]
        processed_hist["rollout/return"] = mean[return_key]
        processed_hist["rollout/return"] = processed_hist["rollout/return"] / 22 * 100
        processed_hist["rollout/return"] = processed_hist["rollout/return"].rolling(rolling_amount, min_periods=1).mean()
        processed_hist["rollout/accumulated_score"] = mean[score_key]

        hist_list.append(pd.DataFrame(processed_hist))

    concatenated = pd.concat(hist_list)

    mean = concatenated.groupby("_step").mean()
    std = concatenated.groupby("_step").std()

    return mean, std