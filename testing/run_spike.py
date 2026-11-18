"""CLI wrapper for a spike workload with a 30/60/30-second default shape."""

import subprocess
import sys
from pathlib import Path


_PHASE_OPTIONS = (
    "--warmup-duration",
    "--peak-duration",
    "--recovery-duration",
)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    has_duration = any(
        value == "--duration" or value.startswith("--duration=")
        for value in arguments
    )
    has_phase_duration = any(
        value == option or value.startswith(option + "=")
        for value in arguments
        for option in _PHASE_OPTIONS
    )
    if not has_duration and not has_phase_duration:
        arguments.extend(
            [
                "--warmup-duration",
                "30",
                "--peak-duration",
                "60",
                "--recovery-duration",
                "30",
            ]
        )
    runner = Path(__file__).with_name("run_load_test.py")
    completed = subprocess.run(
        [sys.executable, str(runner), *arguments, "--scenario", "spike"],
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
