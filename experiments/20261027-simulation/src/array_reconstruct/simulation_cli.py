"""Simulation-only routing for the migrated, unchanged legacy command code."""
from __future__ import annotations

import argparse
import sys


COMMANDS = ("validate", "validate-sparse", "simulate", "reconstruct")


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Migrated synthetic array reconstruction; legacy implementation, not yet optimized.",
        epilog="Use COMMAND --help for the original command options. Run from this experiment directory.",
    )
    parser.add_argument("command", nargs="?", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Original command arguments")
    if not arguments or arguments[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    if arguments[0] not in COMMANDS:
        parser.error("command must be one of: " + ", ".join(COMMANDS))
    from .cli import main as legacy_main

    return legacy_main(arguments)
