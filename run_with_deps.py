from __future__ import annotations

import argparse
import os
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description="Run a Python script with an extra dependency directory.")
    parser.add_argument("--deps", required=True, help="Path to additional dependency directory")
    parser.add_argument("--script", required=True, help="Target Python script path")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to target script")
    args = parser.parse_args()

    deps = str(args.deps or "").strip()
    script = os.path.abspath(str(args.script or "").strip())
    script_dir = os.path.dirname(script)

    if deps:
        sys.path.insert(0, os.path.abspath(deps))
    if script_dir:
        sys.path.insert(0, script_dir)

    forwarded = list(args.script_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    sys.argv = [script] + forwarded
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
