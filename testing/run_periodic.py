"""CLI wrapper for a periodic CloudPulse workload."""

import subprocess
import sys
from pathlib import Path


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    runner = Path(__file__).with_name("run_load_test.py")
    completed = subprocess.run(
        [sys.executable, str(runner), *arguments, "--scenario", "periodic"],
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
