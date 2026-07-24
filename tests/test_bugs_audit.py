"""Reproduce the real bugs found in the deep audit.

Each test is named after the bug it exposes. A FAILING test = bug confirmed.
After fixes, all tests should pass.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

# ---------------------------------------------------------------------------
# BUG 1: inbox_approver drops tool_call_id
# Expected: add_approval is called WITH tool_call_id so durable-resume
#           finds the existing item and does NOT create a duplicate.
# Actual:   tool_call_id is never passed -> for_tool_call() always misses
#           -> a second resume call creates a SECOND inbox item.
# ---------------------------------------------------------------------------

def test_bug1_inbox_approver_drops_tool_call_id(tmp_path):
    """inbox_approver must propagate tool_call_id so durable-resume is idempotent."""
    from coworker.engine import ApprovalOutcome, PermissionRequest
    from coworker.inbox import InboxStore, inbox_approver

    store = InboxStore(tmp_path / "inbox.json")
    approver = inbox_approver(store, "s1")
    req = PermissionRequest(
        tool_name="run_shell",
        arguments={"command": "rm -rf /"},
        metadata=None,
        reason="needs approval",
        tool_call_id="call-abc-123",  # the engine passes this
    )

    async def run():
        async def resolve_soon():
            for _ in range(300):
                pend = store.pending("s1")
                if pend:
                    store.resolve(pend[0].id, "allow")
                    return
                await asyncio.sleep(0.001)

        await asyncio.gather(approver(req), resolve_soon())

    asyncio.run(run())

    items = store.list(session_id="s1")
    assert len(items) == 1, f"Expected 1 item, got {len(items)} (duplicate created)"

    # The item must carry tool_call_id for for_tool_call() to find it on resume
    assert items[0].tool_call_id == "call-abc-123", (
        f"tool_call_id not stored: got {items[0].tool_call_id!r}"
    )

    # Simulating durable resume: calling add() with same tool_call_id must return
    # the EXISTING item (idempotent), not create a new one.
    existing = store.for_tool_call("s1", "call-abc-123")
    assert existing is not None, "for_tool_call() cannot find the item -> resume will re-prompt"


# ---------------------------------------------------------------------------
# BUG 2: InboxStore.wait() race condition
# A resolve() that arrives between the 'already resolved?' check and
# _waiters.setdefault() is silently dropped, leaving wait() blocked forever.
# We simulate this by monkey-patching _waiters.setdefault to inject a
# resolve() call in the window between the check and the setdefault.
# ---------------------------------------------------------------------------

def test_bug2_wait_race_condition_resolve_before_setdefault(tmp_path):
    """resolve() between state-check and setdefault must not cause infinite hang."""
    from coworker.inbox import InboxStore

    store = InboxStore(tmp_path / "inbox.json")
    item = store.add_approval("s1", "Approve?")

    original_setdefault = store._waiters.setdefault

    def racing_setdefault(key, default):
        # Inject resolve() BEFORE the event is registered in _waiters.
        # If the race guard is absent, wait() will block forever after this.
        store.resolve(item.id, "allow")
        return original_setdefault(key, default)

    store._waiters.setdefault = racing_setdefault

    async def run():
        try:
            result = await asyncio.wait_for(store.wait(item.id), timeout=2.0)
            return result
        except asyncio.TimeoutError:
            return "TIMEOUT"  # bug: hung because resolve() was missed

    result = asyncio.run(run())
    assert result != "TIMEOUT", (
        "wait() hung: resolve() that arrived before setdefault() was lost"
    )
    assert result == "allow"


# ---------------------------------------------------------------------------
# BUG 3: InboxStore._save() is not atomic
# write_text() truncates then writes. A crash mid-write corrupts the file.
# We simulate by patching write_text to raise mid-way, then verify the
# file is not left in a truncated/empty state.
# ---------------------------------------------------------------------------

def test_bug3_inbox_save_not_atomic(tmp_path, monkeypatch):
    """_save() must not leave a truncated file if interrupted mid-write."""
    from pathlib import Path
    from coworker.inbox import InboxStore

    store = InboxStore(tmp_path / "inbox.json")
    item = store.add_approval("s1", "Safe data")
    store.resolve(item.id, "allow")

    # Verify the file exists and is valid before the test.
    path = tmp_path / "inbox.json"
    assert path.exists()
    original_content = path.read_text()
    assert "Safe data" in original_content

    # Now simulate a crash mid-write: patch write_text on the Path instance
    # to truncate the file and then raise (simulating disk-full / SIGKILL).
    original_write = Path.write_text

    def crashing_write(self, content, **kwargs):
        if self == path:
            # Truncate first (simulates partial write)
            self.write_bytes(b"")
            raise OSError("Simulated disk full")
        return original_write(self, content, **kwargs)

    monkeypatch.setattr(Path, "write_text", crashing_write)

    # Adding a new item triggers _save() which will crash mid-write
    try:
        store.add_approval("s1", "New item")
    except OSError:
        pass

    # The original data should still be intact if _save() used atomic write.
    # With the bug: the file is now empty -> reload loses all previous items.
    monkeypatch.undo()
    reloaded = InboxStore(tmp_path / "inbox.json")
    items = reloaded.list(session_id="s1")
    assert len(items) >= 1, "All inbox data lost due to non-atomic write"
    assert any(i.title == "Safe data" for i in items), (
        "Previously resolved item wiped by failed non-atomic _save()"
    )


# ---------------------------------------------------------------------------
# BUG 4: shell._bg_counter not thread-safe -> duplicate task IDs
# Concurrent run_background() calls from two threads can produce the same
# task_id, causing the second to silently overwrite the first in _bg_tasks.
# ---------------------------------------------------------------------------

def test_bug4_bg_counter_not_thread_safe(tmp_path):
    """Concurrent run_background() must not produce duplicate task IDs."""
    from coworker.tools.shell import LocalExecutor

    ex = LocalExecutor(cwd=tmp_path, default_timeout=5)
    task_ids = []
    lock = threading.Lock()
    errors = []

    def launch():
        try:
            result = ex.run_background("echo bg")
            tid = result.get("task_id")
            if tid:
                with lock:
                    task_ids.append(tid)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=launch) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ex.close()

    assert not errors, f"Exceptions during concurrent launch: {errors}"
    assert len(task_ids) == len(set(task_ids)), (
        f"Duplicate task IDs found: {sorted(task_ids)}"
    )


# ---------------------------------------------------------------------------
# BUG 5: shell._bg_tasks never pruned -> memory grows unboundedly
# After a background task exits, its entry stays in _bg_tasks forever.
# ---------------------------------------------------------------------------

def test_bug5_bg_tasks_not_pruned(tmp_path):
    """Exited background tasks must not accumulate in _bg_tasks forever."""
    from coworker.tools.shell import LocalExecutor

    ex = LocalExecutor(cwd=tmp_path, default_timeout=5)

    # Start and wait for 15 quick tasks to exit.
    task_ids = []
    for _ in range(15):
        r = ex.run_background("echo done")
        task_ids.append(r["task_id"])

    # Wait for all to exit.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        all_done = all(
            ex._bg_tasks.get(tid) and ex._bg_tasks[tid].proc.poll() is not None
            for tid in task_ids
        )
        if all_done:
            break
        time.sleep(0.1)

    # Read output to trigger any pruning logic (if implemented).
    for tid in task_ids:
        ex.background_output(tid)

    ex.close()

    # After 15 exited tasks, the dict should NOT hold all 15 dead entries.
    # If pruning is absent, this assertion fails proving the leak.
    dead_count = sum(
        1 for t in ex._bg_tasks.values() if t.proc.poll() is not None
    )
    assert dead_count < 15, (
        f"All {dead_count} exited tasks still in _bg_tasks — memory leak confirmed"
    )
