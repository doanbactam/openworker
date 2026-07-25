"""Regression tests for the bugs found in the deep audit.

Meach test is named after the bug it locks down. These used to *reproduce* the
bugs (fail on unpatched code); now the fixes are in (`inbox.py`, `shell.py`) they
assert the correct behavior and must all PASS. A future regression flips one red.

Source of truth:
- coworker/inbox.py       (InboxStore, inbox_approver)
- coworker/engine.py      (PermissionRequest)
- coworker/tools/shell.py (LocalExecutor._bg_counter, _bg_tasks, _BackgroundTask)
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest  # noqa: F401  (kept for future parametrization)


# ---------------------------------------------------------------------------
# BUG 1: inbox_approver must forward tool_call_id
#
# Engine passes tool_call_id into PermissionRequest; inbox_approver must forward
# it to add_approval() so for_tool_call() finds the existing item on durable
# resume instead of creating a second item and re-prompting the user.
# ---------------------------------------------------------------------------

def test_bug1_inbox_approver_forwards_tool_call_id(tmp_path):
    """inbox_approver forwards tool_call_id so durable-resume stays idempotent."""
    from coworker.engine import ApprovalOutcome, PermissionRequest
    from coworker.inbox import InboxStore, inbox_approver

    store = InboxStore(tmp_path / "inbox.json")
    approver = inbox_approver(store, "s1")
    req = PermissionRequest(
        tool_name="run_shell",
        arguments={"command": "rm -rf /"},
        metadata=None,
        reason="needs approval",
        tool_call_id="call-abc-123",
    )

    async def run():
        async def resolve_soon():
            for _ in range(500):
                pend = store.pending("s1")
                if pend:
                    store.resolve(pend[0].id, "allow")
                    return
                await asyncio.sleep(0.001)

        # approver returns an ApprovalOutcome (not a tuple)
        outcome = await asyncio.gather(approver(req), resolve_soon())
        return outcome[0]

    result = asyncio.run(run())
    assert result == ApprovalOutcome.ONCE

    items = store.list(session_id="s1")
    assert len(items) == 1, f"expected 1 item, got {len(items)} (duplicate prompt?)"
    assert items[0].tool_call_id == "call-abc-123", (
        f"tool_call_id not persisted, got {items[0].tool_call_id!r}"
    )

    # Durable resume: for_tool_call must find the existing item, not make a new one.
    existing = store.for_tool_call("s1", "call-abc-123")
    assert existing is not None and existing.id == items[0].id


# ---------------------------------------------------------------------------
# BUG 2: InboxStore.wait() must be race-safe
#
# If resolve() runs between the state-check and _waiters.setdefault(), the newly
# created event is never .set() and wait() would hang forever. The fix registers
# the waiter BEFORE checking resolved state.
# ---------------------------------------------------------------------------

def test_bug2_wait_race_resolve_arrives_before_setdefault(tmp_path):
    """resolve() landing in the setdefault window must not hang wait() forever."""
    from coworker.inbox import InboxStore

    store = InboxStore(tmp_path / "inbox.json")
    item = store.add_approval("s1", "Approve deploy?")

    original_setdefault = store._waiters.setdefault
    injected = False

    def racing_setdefault(key, default):
        nonlocal injected
        if not injected and key == item.id:
            # Resolve right in the race window, before the event is registered.
            store.resolve(item.id, "allow")
            injected = True
        return original_setdefault(key, default)

    store._waiters.setdefault = racing_setdefault

    async def run():
        try:
            return await asyncio.wait_for(store.wait(item.id), timeout=2.0)
        except asyncio.TimeoutError:
            return "TIMEOUT"

    result = asyncio.run(run())
    assert result != "TIMEOUT", (
        "wait() hung: resolve() in the setdefault window left the event unset"
    )
    assert result == "allow", f"expected 'allow', got {result!r}"


# ---------------------------------------------------------------------------
# BUG 3: InboxStore._save() must be atomic + _load() must tolerate corruption
#
# Non-atomic write_text() truncates then writes; a crash mid-write left an empty
# file and lost all data. The fix writes to <path>.tmp then os.replace()s it, so
# a crash before the replace leaves the original file fully intact.
# ---------------------------------------------------------------------------

def test_bug3_save_is_atomic_original_survives_crash(tmp_path, monkeypatch):
    """A crash while writing the .tmp file must not touch the original inbox."""
    from pathlib import Path
    from coworker.inbox import InboxStore

    path = tmp_path / "inbox.json"
    store = InboxStore(path)
    item = store.add_approval("s1", "Safe data")
    store.resolve(item.id, "allow")

    assert path.exists() and "Safe data" in path.read_text()

    original_write_text = Path.write_text

    def crashing_write(self, content, *args, **kwargs):
        # Atomic save writes to <path>.tmp first; simulate a crash there.
        if self.suffix == ".tmp":
            self.write_bytes(b'{"items": [garbage')  # tmp left half-written
            raise OSError("Simulated disk full mid-write")
        return original_write_text(self, content, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", crashing_write)
    try:
        store.add_approval("s1", "New item triggers save")
    except OSError:
        pass  # expected
    monkeypatch.undo()

    # os.replace() never ran, so the original file must still hold the old data.
    reloaded = InboxStore(path)
    items = reloaded.list(session_id="s1")
    assert any(i.title == "Safe data" for i in items), (
        "atomic save failed: 'Safe data' lost after a crash mid-write"
    )


def test_bug3_load_tolerates_corrupt_file(tmp_path):
    """_load() must not crash on an empty/truncated/corrupt inbox file."""
    from coworker.inbox import InboxStore

    path = tmp_path / "inbox.json"

    path.write_text("")  # truncated (what the old non-atomic write could leave)
    assert InboxStore(path).list() == []

    path.write_text("{ this is not valid json")  # corrupt
    assert InboxStore(path).list() == []


# ---------------------------------------------------------------------------
# BUG 4: LocalExecutor bg counter must be thread-safe
#
# Concurrent run_background() calls must not read the same counter value and
# produce duplicate task IDs (which would overwrite each other in _bg_tasks).
# ---------------------------------------------------------------------------

def test_bug4_bg_counter_is_thread_safe(tmp_path):
    """Concurrent run_background() must produce unique task IDs."""
    from coworker.tools.shell import LocalExecutor

    ex = LocalExecutor(cwd=tmp_path, default_timeout=5)
    task_ids: list[str] = []
    id_lock = threading.Lock()
    errors: list[str] = []

    def launch():
        try:
            result = ex.run_background("echo bg")
            tid = result.get("task_id")
            if tid:
                with id_lock:
                    task_ids.append(tid)
        except Exception as e:  # noqa: BLE001
            with id_lock:
                errors.append(str(e))

    threads = [threading.Thread(target=launch) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ex.close()

    assert not errors, f"exceptions during launch: {errors}"
    assert len(task_ids) == 20
    assert len(set(task_ids)) == len(task_ids), (
        f"duplicate task IDs: {sorted(task_ids)}"
    )


# ---------------------------------------------------------------------------
# BUG 5: exited bg tasks must be pruned from _bg_tasks (no memory leak)
#
# background_output() prunes a task once it has fully finished (process exited
# AND reader thread drained) and all its output has been read. The first read
# returns the output; a later read (buffer empty) triggers the prune.
# ---------------------------------------------------------------------------

def test_bug5_dead_bg_tasks_are_pruned(tmp_path):
    """Exited-and-drained background tasks must be removed from _bg_tasks."""
    from coworker.tools.shell import LocalExecutor

    ex = LocalExecutor(cwd=tmp_path, default_timeout=5)
    N = 20
    task_ids = [ex.run_background("echo done")["task_id"] for _ in range(N)]

    # Wait until every process has exited and its reader thread has drained.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if all(
            tid not in ex._bg_tasks or ex._bg_tasks[tid].is_finished()
            for tid in task_ids
        ):
            break
        time.sleep(0.05)

    # Drain: read repeatedly until each finished task gets pruned. The first read
    # delivers "done", a later read (empty buffer) triggers the prune.
    for tid in task_ids:
        for _ in range(10):
            if tid not in ex._bg_tasks:
                break
            ex.background_output(tid)
            time.sleep(0.02)

    ex.close()

    leftover = [tid for tid in task_ids if tid in ex._bg_tasks]
    assert leftover == [], (
        f"prune failed: {len(leftover)}/{N} exited tasks still in _bg_tasks: {leftover}"
    )
