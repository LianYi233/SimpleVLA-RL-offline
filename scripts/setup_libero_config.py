#!/usr/bin/env python3
"""Write ~/.libero/config.yaml for AutoDL / custom LIBERO layouts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

LIBERO_SUITES = ("libero_10", "libero_90", "libero_spatial", "libero_object", "libero_goal", "libero_100")


def has_suite_dirs(path: Path) -> bool:
    return any((path / suite).is_dir() for suite in LIBERO_SUITES)


def has_hdf5(path: Path) -> bool:
    return any(path.rglob("*.hdf5"))


def resolve_datasets_root(upload_root: Path) -> Path:
    upload_root = upload_root.resolve()
    candidates = [
        upload_root,
        upload_root / "datasets",
        upload_root / "libero" / "datasets",
    ]
    for candidate in candidates:
        if candidate.is_dir() and (has_suite_dirs(candidate) or has_hdf5(candidate)):
            return candidate
    raise FileNotFoundError(
        f"Could not find LIBERO demo folders under {upload_root}. "
        f"Expected subdirs like libero_10/ with *.hdf5 files."
    )


def resolve_libero_repo(explicit: str | None) -> Path:
    if explicit:
        repo = Path(explicit).resolve()
        if (repo / "libero" / "libero").is_dir():
            return repo
        raise FileNotFoundError(f"Invalid LIBERO repo: {repo}")

    candidates = [
        Path("/root/2604-VLA_RL_offline/SimpleVLA_RL_Offline-main/LIBERO"),
        Path("/root/autodl-tmp/LIBERO"),
        Path.cwd() / "LIBERO",
    ]
    for candidate in candidates:
        if (candidate / "libero" / "libero").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate LIBERO code repo. Pass --libero-repo /path/to/LIBERO"
    )


def build_config(libero_repo: Path, datasets_root: Path) -> dict[str, str]:
    base = libero_repo / "libero" / "libero"
    return {
        "assets": str(base / "assets"),
        "bddl_files": str(base / "bddl_files"),
        "benchmark_root": str(base),
        "datasets": str(datasets_root),
        "init_states": str(base / "init_files"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup LIBERO config.yaml")
    parser.add_argument(
        "--datasets-root",
        default=os.environ.get("LIBERO_DATASETS_ROOT", "/root/autodl-tmp/datasets/LIBERO"),
        help="Uploaded LIBERO dataset directory",
    )
    parser.add_argument(
        "--libero-repo",
        default=os.environ.get("LIBERO_ROOT"),
        help="LIBERO git repo root (contains libero/libero/)",
    )
    parser.add_argument(
        "--config-path",
        default=os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")),
        help="Directory to write config.yaml",
    )
    args = parser.parse_args()

    libero_repo = resolve_libero_repo(args.libero_repo)
    datasets_root = resolve_datasets_root(Path(args.datasets_root))
    config = build_config(libero_repo, datasets_root)

    config_dir = Path(args.config_path)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    with config_file.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Wrote {config_file}")
    for key, value in config.items():
        print(f"  {key}: {value}")

    missing = [key for key, value in config.items() if key != "datasets" and not Path(value).exists()]
    if missing:
        print("\nWarning: some LIBERO code paths are missing:")
        for key in missing:
            print(f"  - {key}: {config[key]}")
        print("Install LIBERO code with: pip install -e /path/to/LIBERO")


if __name__ == "__main__":
    main()
