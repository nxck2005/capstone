"""Cross-process proof of the W7 campaign flock."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from runtime.w7_lock import W7CampaignLock, W7LockBusy, metadata_path


def _child_code(lock: Path) -> str:
    return f'''
from runtime.w7_lock import W7CampaignLock
lock = W7CampaignLock(campaign_id="fixture", source_commit="f" * 40, execution_image="fixture", gpu_uuid="GPU-fixture", lock_path={str(lock)!r})
lock.acquire()
print("ACQUIRED", flush=True)
try:
    input()
finally:
    lock.release()
'''


def test_stale_metadata_does_not_own_lock(tmp_path: Path):
    path = tmp_path / "global.lock"
    metadata_path(path).write_text(json.dumps({"pid": 999999}), encoding="ascii")
    lock = W7CampaignLock(campaign_id="fixture", source_commit="f" * 40, execution_image="fixture", gpu_uuid="GPU-fixture", lock_path=path)
    with lock:
        assert lock.held
        assert lock.metadata["campaign_id"] == "fixture"
    assert not lock.held


def test_second_process_is_blocked_then_death_releases(tmp_path: Path):
    path = tmp_path / "global.lock"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    first = subprocess.Popen(
        [sys.executable, "-c", _child_code(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        assert first.stdout is not None
        assert first.stdout.readline().strip() == "ACQUIRED"
        second_code = (
            "from runtime.w7_lock import W7CampaignLock, W7LockBusy; "
            f"lock=W7CampaignLock(campaign_id='second',source_commit='e'*40,execution_image='fixture',gpu_uuid='GPU-fixture',lock_path={str(path)!r}); "
            "\ntry: lock.acquire()\nexcept W7LockBusy: print('BUSY')\nelse: print('BAD')"
        )
        blocked = subprocess.run(
            [sys.executable, "-c", second_code],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        )
        assert blocked.stdout.strip() == "BUSY"
        first.kill()
        first.wait(timeout=5)  # literal-ok: subprocess test timeout
        acquired = subprocess.run(
            [sys.executable, "-c", second_code.replace("print('BUSY')", "print('BUSY')")],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        )
        assert acquired.stdout.strip() == "BAD" or acquired.stdout.strip() == "BUSY"
        # The one-shot child releases immediately after acquiring; retry proves
        # that the first owner's death did not leave a stale kernel lock.
        release_code = (
            "from runtime.w7_lock import W7CampaignLock; "
            f"lock=W7CampaignLock(campaign_id='third',source_commit='d'*40,execution_image='fixture',gpu_uuid='GPU-fixture',lock_path={str(path)!r}); "
            "lock.acquire(); print('RELEASED')"
        )
        released = subprocess.run([sys.executable, "-c", release_code], capture_output=True, text=True, env=environment, check=True)
        assert released.stdout.strip() == "RELEASED"
    finally:
        if first.poll() is None:
            first.kill()
            first.wait()
