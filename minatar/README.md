# Identifiable Token Correspondence for World Models

## Setup

Use Python 3.10. Install the necessary packages from `requirements.txt`:
```
pip install -r requirements.txt
```

## Configuration

Configuration files are stored in the `config/` directory. You can create your own configuration file or pass additional configs as command line arguments directly to the scripts below. Available options are listed in `configs.py`.

## Policy training

A training run can be launched with

```
python train.py --config_path={config_path} [--{additional_configs} ...]
```

Example:
```
python train.py --config_path=config/minatar_asterix_itc.yaml --seed=1
```

### Configs
* ITC: `config/minatar_{game}_itc.yaml`
* Baseline (reproduced): `config/minatar_{game}_baseline.yaml`

## Evaluation

You can evaluate the policy of the checkpoints with `evaluate_minatar.py`. Set `restore_ckpt_path` (required) and `restore_ckpt_step` (optional) to evaluate. Correct configuration for the checkpoint is needed for accurate evaluation. You should modify `filename` variable (L71) inside the code to save the evaluation result.

```
python evaluate_minatar.py --config_path={config_path} --restore_ckpt_path={checkpoint_path} --restore_ckpt_step={checkpoint_step}
```

## Graphs

You can generate a graph of returns with `compare_minatar_return.py`. You should set paths for the results inside the code. It requires all evaluations from `ckpt_steps` (L92), which will be saved by default when you run the training.

Then run
```
python compare_minatar_return.py
```
