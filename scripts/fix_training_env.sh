#!/usr/bin/env bash
# Fix common dependency issues before running offline GRPO training.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-simplevla}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/2604-VLA_RL_offline/SimpleVLA_RL_Offline-main}"
VERL_ROOT="${VERL_ROOT:-$(python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("verl")
print(spec.submodule_search_locations[0] if spec and spec.submodule_search_locations else "")
PY
)}"

echo "=== Fixing training environment (conda env: ${CONDA_ENV}) ==="

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

echo "[1/4] Pinning peft for transformers 4.40.x compatibility..."
pip install 'peft==0.11.1'

echo "[2/4] Installing missing tokenizer/runtime deps if needed..."
pip install sentencepiece tiktoken timm

echo "[3/4] Restoring broken flash_attn optional-import patches (if any)..."
python3 - <<'PY'
from pathlib import Path

FLASH_IMPORT = "from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis"
OPTIONAL_BLOCK = """try:
    from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
except ImportError:
    pad_input = unpad_input = rearrange = index_first_axis = None"""

def fix_file(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text()
    original = text

    # Remove accidental double try/except blocks from prior manual patches.
    while "try:\n    try:" in text:
        text = text.replace("try:\n    try:", "try:")

    if FLASH_IMPORT in text and OPTIONAL_BLOCK not in text:
        text = text.replace(FLASH_IMPORT, OPTIONAL_BLOCK)

    if text != original:
        path.write_text(text)
        print(f"  patched: {path}")

project_root = Path(__import__("os").environ.get("PROJECT_ROOT", "/root/2604-VLA_RL_offline/SimpleVLA_RL_Offline-main"))
verl_root = Path(__import__("os").environ.get("VERL_ROOT", ""))

candidates = [
    project_root / "src" / "dp_rob.py",
]
if verl_root:
    candidates.extend([
        Path(verl_root) / "workers" / "actor" / "dp_actor.py",
        Path(verl_root) / "workers" / "actor" / "dp_prime.py",
    ])

for candidate in candidates:
    fix_file(candidate)
PY

echo "[4/4] Verifying imports..."
python3 - <<'PY'
import peft
import transformers
print(f"peft={peft.__version__}, transformers={transformers.__version__}")
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
print("peft import OK")
PY

cat <<'EOF'

Environment fix complete.

If training still fails on flash_attn, either:
  1) keep use_remove_padding=False (default in offline GRPO script), or
  2) install flash-attn:
     pip install packaging ninja
     pip install flash-attn --no-build-isolation

Then restart training:
  tmux attach -t simple
  bash src/examples/run_openvla_oft_rl_libero_offline_grpo.sh
EOF
