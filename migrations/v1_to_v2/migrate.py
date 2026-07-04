#!/usr/bin/env python3
"""TLBrain v1 → v2 migration entry point.

Usage:
    python -m migrations.v1_to_v2.migrate [--dry-run]
"""
import argparse
import logging
import os
import sys
from pathlib import Path


def _load_env() -> None:
    env_file = Path(__file__).parents[2] / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env()

from migrations.v1_to_v2.steps import (
    step_build_folders,
    step_delete_clients,
    step_migrate_qdrant,
    step_update_transcript_index,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STEPS = [
    ("1: build folders/", step_build_folders),
    ("2: update transcript_index", step_update_transcript_index),
    ("3: migrate Qdrant payload", step_migrate_qdrant),
    ("4: delete clients/", step_delete_clients),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="TLBrain v1 → v2 data migration")
    parser.add_argument("--dry-run", action="store_true", help="Log changes without writing")
    args = parser.parse_args()

    dry_run = args.dry_run
    if dry_run:
        logger.info("=== DRY RUN — no writes will be made ===")

    for name, fn in STEPS:
        logger.info("--- Step %s ---", name)
        result = fn(dry_run=dry_run)
        logger.info("Result: %s", result)

    if not dry_run:
        logger.info("Migration complete. Run: python -m migrations.v1_to_v2.verify")
    else:
        logger.info("Dry run complete.")


if __name__ == "__main__":
    main()
