set -euo pipefail

pip install uv
uv pip install -r requirements.txt
uv pip install -e ./libs/internal
uv pip install -e ./libs/external

python scripts/make-dataset/bar.py
python scripts/make-dataset/cdc-diabetes.py
python scripts/make-dataset/diabetes-130.py
python scripts/make-dataset/mushroom.py
