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
python train.py --config_path=config/itc.yaml --seed=1
```

### Configs
* ITC: `config/itc.yaml`
* Baseline (reproduced): `config/baseline_reproduced.yaml`
* Baseline (our hyperparameters): `config/baseline_our_hyperparams.yaml`
* ITC on Craftax full: `config/full_craftax_env.yaml`

## World model evaluation

### Prediction accuracy

To measure the world model prediction accuracy of a saved model, follow these steps. First, modify `generate_model_outputs_dataset.py` to choose a save directory. Then, run
```
python generate_model_outputs_dataset.py --config_path={config_path} --restore_ckpt_path={restore_ckpt_path} --dataset_seed=0
```
where `config_path` and `restore_ckpt_path` correspond to the config and the checkpoint directory of your saved model. This script will create a dataset of environment transitions. You can run the command multiple times with different `dataset_seed` to create a range of multiple datasets (to avoid out-of-memory issues if trying to create a large dataset).

To measure the accuracy of a saved model of our method (ITC), use `measure_optimal_transport_accuracy.py`. First, modify the file to point to your dataset directory saved above, set the range of datasets depending on which `dataset_seed`(s) you used, and set the checkpoint directory of the saved model. Then run
```
python measure_optimal_transport_accuracy.py --config_path={config_path} --restore_ckpt_path={restore_ckpt_path}
```
The script will print the accuracy result after applying optimal transport ("Current model matches") and ("Comparison matches").

To measure the accuracy of a saved baseline model, use `measure_transformer_accuracy.py`. First, modify the file to point to your dataset directory saved above and set the range of datasets depending on which `dataset_seed`(s) you used. Then run
```
python measure_transformer_accuracy.py --config_path={config_path} --restore_ckpt_path={restore_ckpt_path}
```

### Visualizing imagination rollouts

You need to change `restore_ckpt_path` in Line 22 manually for the baseline model.
The `restore_ckpt_path` passed with arguments should be the model trained with spatio-temporal PE and optimal transport decoding.

```
python compare_imagination.py --config_path={config_path} --restore_ckpt_path={restore_ckpt_path}
```

### Configs
* Baseline: `config/baseline_our_hyperparams.yaml`
* ITC: `config/itc.yaml`

## Graphs

You can generate a graph of returns with `compare_return.py`. First, modify the file to list your recorded training runs from WANDB. The format should look like
```
    "team_name/project_name/41iw4nva",
```
where 41iw4nva is a run ID.

Then run
```
python compare_return.py
```