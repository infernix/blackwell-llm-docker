#!/usr/bin/env python3
"""Wait for backend health while failing immediately if its container exits."""
import argparse
import subprocess
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--container", required=True)
p.add_argument("--url", required=True)
p.add_argument("--timeout", type=float, default=1200)
a = p.parse_args()
deadline = time.monotonic() + a.timeout
while time.monotonic() < deadline:
    status = subprocess.run(
        ["podman", "inspect", "-f", "{{.State.Running}} {{.State.ExitCode}}", a.container],
        capture_output=True,
        text=True,
    )
    if status.returncode or not status.stdout.startswith("true "):
        raise SystemExit(f"container failed before readiness: {status.stdout.strip()}")
    try:
        with urllib.request.urlopen(a.url, timeout=1) as response:
            if response.status == 200:
                print("health=200")
                raise SystemExit(0)
    except (urllib.error.URLError, TimeoutError):
        pass
    time.sleep(0.2)
raise SystemExit("health timeout")
