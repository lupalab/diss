# Dynamic Information Sub-Selection for Adaptive Decision Support

This repository contains the source code and prepared data for the paper "Dynamic Information Sub-Selection for Adaptive Decision Support." The code trains and evaluates adaptive feature-subselection policies for decision support settings where a system chooses which information to forward to a downstream decision maker.

The main experiment implementations are in `experiments/`:

- `experiments/classic-bandits`: adaptive DISS experiments with classic acquisition strategies and Mimic reward estimators.
- `experiments/modiste`: MODISTE baseline experiments.
- `experiments/all-features`: all-feature baseline that forwards every feature.
- `experiments/template`: a minimal Hydra training template.

## Setup

Create and activate a Python environment, then install dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
bash setup.sh
```

`setup.sh` installs `uv`, installs `requirements.txt`, installs the local packages in `libs/internal` and `libs/external`, and runs the public dataset preparation scripts in `scripts/make-dataset/`.

Equivalent manual setup:

```bash
pip install uv
uv pip install -r requirements.txt
uv pip install -e ./libs/internal
uv pip install -e ./libs/external
python scripts/make-dataset/bar.py
python scripts/make-dataset/cdc-diabetes.py
python scripts/make-dataset/diabetes-130.py
python scripts/make-dataset/mushroom.py
```

The generated and packaged public datasets are stored under `data/uci/`. OAI data is not bundled; access must be requested from the Osteoarthritis Initiative and configured locally before running OAI experiments.

## Data

The paper uses the following datasets:

| Dataset | Config name | Notes |
| --- | --- | --- |
| Bar Crawl | `bar-crawl` | Heavy drinking detection from smartphone accelerometer features. |
| CDC Diabetes | `cdc-diab` | UCI Diabetes Health Indicators binary classification task. |
| Diabetes 130-US Hospitals | `diab130` | UCI hospital readmission task collapsed to binary readmission. |
| Secondary Mushroom | `secondary-mushroom` | UCI mushroom edibility classification task. |
| OAI KLG | `oai-klg` | Osteoarthritis Initiative KLG severity task; data requires external access. |

## Running Experiments

All experiment scripts use Hydra. Run commands from inside the corresponding experiment directory so relative config paths and output paths resolve correctly.

Hydra outputs are written under each experiment's `outputs/` directory by default, using names configured in the YAML files, for example `outputs/bar-crawl-overload/<timestamp>/`.

### DISS and Classic Bandit Baselines

Use `experiments/classic-bandits/train.py` for the main adaptive feature-subselection experiments. The primary Hydra axes are:

- `-cp=conf/<dataset>`: dataset/environment config directory.
- `-cn=<environment>`: environment config, such as `overload`, `simplicity`, `multi4`, `interp`, or `vllm` when available.
- `+strat=<strategy>`: acquisition strategy, such as `ts`, `random`, `ei`, `mtspm`, or `revi`.
- `+reward_est=<estimator>`: reward estimator, such as `xgb-5-64`, `structure-5-64-64`, or `structure-zero-one-5-64-64`.
- `+make_envs_func.vdata_seed='range(5)'`: sweep over validation-data seeds.
- `train_conf.n_iter=4002`: set the total interaction/query budget.

Example: run Mimic/structured reward estimation with Thompson sampling on Bar Crawl cognitive-overload environments:

```bash
cd experiments/classic-bandits
CUDA_VISIBLE_DEVICES="" python train.py -m \
  -cp=conf/bar-crawl \
  -cn=overload \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=18 \
  +make_envs_func.vdata_seed='range(5)' \
  +strat=ts \
  +reward_est=structure-5-64-64 \
  train_conf.n_iter=4002
```

Example: run non-Mimic classic acquisition baselines with an XGBoost reward estimator:

```bash
cd experiments/classic-bandits
CUDA_VISIBLE_DEVICES="" python train.py -m \
  -cp=conf/bar-crawl \
  -cn=overload \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=36 \
  +make_envs_func.vdata_seed='range(5)' \
  +reward_est=xgb-5-64 \
  +strat='choice(random,ei,mtspm,revi)' \
  train_conf.n_iter=4002
```

Common dataset/environment combinations used in the paper include:

```bash
# Cognitive overload and simplicity settings
-cp=conf/bar-crawl -cn=overload
-cp=conf/bar-crawl -cn=simplicity
-cp=conf/cdc-diab -cn=overload
-cp=conf/cdc-diab -cn=simplicity
-cp=conf/diab130 -cn=overload
-cp=conf/diab130 -cn=simplicity
-cp=conf/secondary-mushroom -cn=overload
-cp=conf/secondary-mushroom -cn=simplicity
-cp=conf/oai-klg -cn=overload
-cp=conf/oai-klg -cn=simplicity

# vLLM/LLM decision-support settings, where configured
-cp=conf/bar-crawl -cn=vllm
-cp=conf/cdc-diab -cn=vllm
-cp=conf/diab130 -cn=vllm
-cp=conf/secondary-mushroom -cn=vllm
-cp=conf/oai-klg -cn=vllm
```

The complete historical command list for this experiment family was derived from `experiments/sequential-bandits/tmp/cmd.txt` in the development workspace.

### MODISTE Baseline

Use `experiments/modiste/train.py` for MODISTE runs. MODISTE uses `+strat=modiste` and the `unified-knn-25` reward estimator.

Example:

```bash
cd experiments/modiste
CUDA_VISIBLE_DEVICES="" python train.py -m \
  -cp=conf/adni \
  -cn=overload \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=5 \
  +make_envs_func.vdata_seed='range(5)' \
  +strat=modiste \
  +reward_est=unified-knn-25 \
  train_conf.n_iter=4001
```

For vLLM environments, provide a local OpenAI-compatible endpoint and API key through Hydra overrides rather than hard-coding credentials:

```bash
cd experiments/modiste
CUDA_VISIBLE_DEVICES="" python train.py -m \
  -cp=conf/oai-klg \
  -cn=vllm \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=50 \
  +make_envs_func.vdata_seed='range(50)' \
  +strat=modiste \
  +reward_est=unified-knn-25 \
  make_envs_func.server_url="$OPENAI_BASE_URL" \
  make_envs_func.api_key="$OPENAI_API_KEY"
```

### All-Feature Baseline

Use `experiments/all-features/train.py` to evaluate the policy that forwards all features.

Example:

```bash
cd experiments/all-features
python train.py -m -cp=conf/bar-crawl -cn=overload hydra/launcher=joblib
```

A representative sweep over public datasets is:

```bash
cd experiments/all-features
python train.py -m -cp=conf/bar-crawl -cn=overload hydra/launcher=joblib
python train.py -m -cp=conf/bar-crawl -cn=simplicity hydra/launcher=joblib
python train.py -m -cp=conf/bar-crawl -cn=vllm hydra/launcher=joblib
python train.py -m -cp=conf/cdc-diab -cn=overload hydra/launcher=joblib
python train.py -m -cp=conf/cdc-diab -cn=simplicity hydra/launcher=joblib
python train.py -m -cp=conf/cdc-diab -cn=vllm hydra/launcher=joblib
python train.py -m -cp=conf/diab130 -cn=overload hydra/launcher=joblib
python train.py -m -cp=conf/diab130 -cn=simplicity hydra/launcher=joblib
python train.py -m -cp=conf/diab130 -cn=vllm hydra/launcher=joblib
python train.py -m -cp=conf/secondary-mushroom -cn=overload hydra/launcher=joblib
python train.py -m -cp=conf/secondary-mushroom -cn=simplicity hydra/launcher=joblib
python train.py -m -cp=conf/secondary-mushroom -cn=vllm hydra/launcher=joblib
```

## Post-Processing

`experiments/classic-bandits` includes utilities for analyzing trained adaptive policies. These scripts take a prior training output path via `+train_exp.exp_p=...` and one or more run IDs via `+train_exp.run_id=...`.

Distill a trained policy into a decision tree:

```bash
cd experiments/classic-bandits
python distill-tree.py -m \
  -cp=conf \
  -cn=distill \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=20 \
  n_clusters='range(2,11,4)' \
  +train_exp.exp_p=experiments/classic-bandits/outputs/bar-crawl-simplicity/<timestamp> \
  +train_exp.run_id='range(0,50)' \
  hydra.job.name=bar-simplicity-distill-ksweep
```

Compute a best static feature mask for trained runs:

```bash
cd experiments/classic-bandits
python best-static-mask.py -m \
  -cp=conf \
  -cn=best-static \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=20 \
  +train_exp.exp_p=experiments/classic-bandits/outputs/bar-crawl-simplicity/<timestamp> \
  +train_exp.run_id='range(0,50)' \
  hydra.job.name=bar-simplicity-best-static
```

Gather selected actions/responses from trained runs:

```bash
cd experiments/classic-bandits
python gather-responses.py -m \
  -cp=conf \
  -cn=gather-acts \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=20 \
  +train_exp.exp_p=experiments/classic-bandits/outputs/bar-crawl-simplicity/<timestamp> \
  +train_exp.run_id='range(0,50)' \
  hydra.job.name=bar-simplicity-gather-actions
```

Replace `<timestamp>` with the timestamp directory created by the corresponding training sweep.

## Reproducibility Notes

The paper experiments used an initial random acquisition budget of 500 expert queries and a total budget of 4000 queries. Many reproduced commands therefore override `train_conf.n_iter` to `4001` or `4002`, depending on whether the script counts the initial query step separately.

Large sweeps use Hydra's Joblib launcher. Adjust `hydra.launcher.n_jobs` to match available CPU cores and memory. The paper reports experiments run on a Lenovo ThinkStation P520 with an Intel Xeon W-2295 and 512 GB RAM.

For CPU-only execution, the historical commands set:

```bash
CUDA_VISIBLE_DEVICES=""
```

For LLM/vLLM decision-support settings, configure the server URL and API key via environment variables or Hydra overrides. Do not commit private endpoint URLs or API keys.

To launch a local MedGemma vLLM server, follow the vLLM documentation to install vLLM, then start an OpenAI-compatible server, for example:

```bash
vllm serve google/medgemma-1.5-4b-it \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --h11-max-header-count 9999999 \
  --limit-mm-per-prompt '{"image": 0}'
```

The exact vLLM configuration requires manual tuning based on the GPU model, available memory, and other workloads running on the machine. **We recommend installing vllm in its own python virtual environment to avoid any kind of dependency conflicts**.

## Citation

If you use this code, cite the accompanying paper:

```bibtex
@inproceedings{huang2026dynamic,
  title = {Dynamic Information Sub-Selection for Adaptive Decision Support},
  author = {Huang et al.},
  booktitle = {ACM BCB},
  year = {2026}
}
```
