#!/usr/bin/env bash
# Apply offline GRPO reward fixes to a local SimpleVLA-RL-offline checkout.
# Usage:
#   bash apply_offline_reward_fix.sh
#   bash apply_offline_reward_fix.sh /path/to/SimpleVLA_RL_Offline-main

set -euo pipefail

TARGET_DIR="${1:-/root/2604-VLA_RL_offline/SimpleVLA_RL_Offline-main}"
PATCH_URL="https://github.com/LianYi233/SimpleVLA-RL-offline/compare/main...cursor/fix-offline-reward-grpo-9c89.patch"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$TARGET_DIR/src" ]]; then
  echo "Error: target directory not found or missing src/: $TARGET_DIR" >&2
  exit 1
fi

cd "$TARGET_DIR"

apply_patch() {
  local patch_file="$1"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git apply --check "$patch_file"
    git apply "$patch_file"
    echo "Applied patch with git apply."
  else
    patch -p1 --dry-run < "$patch_file"
    patch -p1 < "$patch_file"
    echo "Applied patch with patch -p1."
  fi
}

if [[ -f "$SCRIPT_DIR/fix-offline-reward-grpo.patch" ]]; then
  echo "Using local patch: $SCRIPT_DIR/fix-offline-reward-grpo.patch"
  apply_patch "$SCRIPT_DIR/fix-offline-reward-grpo.patch"
elif command -v curl >/dev/null 2>&1; then
  echo "Downloading patch from GitHub..."
  tmp_patch="$(mktemp)"
  curl -fsSL "$PATCH_URL" -o "$tmp_patch"
  apply_patch "$tmp_patch"
  rm -f "$tmp_patch"
else
  echo "Error: no local patch found and curl unavailable." >&2
  exit 1
fi

echo "Done. Updated files:"
echo "  - src/rob_rollout.py"
echo "  - src/core_algos.py"
echo "  - src/main_ppo.py"
echo "  - src/ray_trainer.py"
echo "  - src/config/ppo_trainer.yaml"
