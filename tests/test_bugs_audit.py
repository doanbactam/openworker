"""Reproduce the real bugs found in deep audit.

Mỗi test tên theo bug nó expose. Test FAIL = bug confirmed tồn tại.
Sau khi fix, tất cả phải pass.

Đọc source thực tế trước khi viết:
- coworker/inbox.py  (InboxStore, inbox_approver)
- coworker/engine.py (PermissionRequest)
- coworker/tools/shell.py (LocalExecutor._bg_counter, _bg_tasks)
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest


# ---------------------------------------------------------------------------
# BUG 1: inbox_approver bỏ tool_call_id
#
# Source inbox_approver (inbox.py, cuối file):
#
#   async def approve(request):
#       item = store.add_approval(
#           session_id,
#           title=f"Run `{request.tool_name}`?",
#           body=request.reason or "",
#           inbox=inbox,
#           # tool_call_id=request.tool_call_id  <-- MISSING
#       )
#
# Engine (engine.py) truyền tool_call_id vào PermissionRequest nhưng
# inbox_approver không forward nó sang add_approval → for_tool_call() trả None
# → durable resume tạo item thứ 2 → user bị hỏi lại lần nữa.
# ---------------------------------------------------------------------------

def test_bug1_inbox_approver_drops_tool_call_id(tmp_path):
    """inbox_approver phải forward tool_call_id để durable-resume idempotent."""
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

        # approver trả ApprovalOutcome (không phải tuple)
        outcome = await asyncio.gather(approver(req), resolve_soon())
        return outcome[0]  # outcome[1] là None từ resolve_soon

    asyncio.run(run())

    items = store.list(session_id="s1")
    assert len(items) == 1, (
        f"BUG 1 CONFIRMED: tạo {len(items)} items thay vì 1 "
        "(duplicate do tool_call_id không được lưu)"
    )

    # Item phải mang tool_call_id để for_tool_call() tìm được khi resume
    assert items[0].tool_call_id == "call-abc-123", (
        f"BUG 1 CONFIRMED: tool_call_id không lưu, got {items[0].tool_call_id!r}\n"
        "Consequence: durable resume gọi add_approval lần 2 → item mới → prompt lại"
    )

    # Simulate durable resume: for_tool_call phải tìm được item cũ
    existing = store.for_tool_call("s1", "call-abc-123")
    assert existing is not None, (
        "BUG 1 CONFIRMED: for_tool_call() trả None → resume sẽ tạo duplicate item"
    )


# ---------------------------------------------------------------------------
# BUG 2: InboxStore.wait() race condition
#
# Source wait() (inbox.py):
#
#   async def wait(self, item_id):
#       item = self._items.get(item_id)
#       if item and item.state == STATE_RESOLVED:   # <-- check state
#           return item.resolution or ""
#       ev = self._waiters.setdefault(item_id, asyncio.Event())  # <-- race window
#       await ev.wait()
#
# Nếu resolve() chạy giữa dòng check và setdefault → event được tạo
# nhưng resolve() đã fire rồi → .set() không gọi nữa → wait() block mãi mãi.
# ---------------------------------------------------------------------------

def test_bug2_wait_race_resolve_arrives_before_setdefault(tmp_path):
    """resolve() đến giữa state-check và setdefault phải không gây hang vĩnh viễn."""
    from coworker.inbox import InboxStore

    store = InboxStore(tmp_path / "inbox.json")
    item = store.add_approval("s1", "Approve deploy?")

    # Patch setdefault để inject resolve() vào đúng race window
    original_setdefault = store._waiters.setdefault

    injected = False

    def racing_setdefault(key, default):
        nonlocal injected
        if not injected and key == item.id:
            # Resolve TRƯỚC KHI event được đăng ký vào _waiters
            # Nếu bug tồn tại: resolve() gọi _waiters.get() → None → không set()
            # → event mới tạo xong nhưng không bao giờ được set → hang mãi mãi
            store.resolve(item.id, "allow")
            injected = True
        return original_setdefault(key, default)

    store._waiters.setdefault = racing_setdefault

    async def run():
        try:
            result = await asyncio.wait_for(store.wait(item.id), timeout=2.0)
            return result
        except asyncio.TimeoutError:
            return "TIMEOUT"

    result = asyncio.run(run())
    assert result != "TIMEOUT", (
        "BUG 2 CONFIRMED: wait() hung sau 2 giây.\n"
        "resolve() chạy trước setdefault() → event không bao giờ được .set()\n"
        "Fix: check lại state SAU setdefault() hoặc dùng lock bao phủ cả 2 bước"
    )
    assert result == "allow", f"Expected 'allow', got {result!r}"


# ---------------------------------------------------------------------------
# BUG 3: InboxStore._save() không atomic
#
# Source _save() (inbox.py):
#
#   def _save(self):
#       self.path.write_text(json.dumps(...), encoding="utf-8")
#
# write_text() = truncate → write. Crash giữa chừng (disk full, SIGKILL)
# → file bị truncate về rỗng → reload mất toàn bộ data.
# Fix: ghi ra .tmp rồi os.replace() (atomic trên POSIX và Windows).
# ---------------------------------------------------------------------------

def test_bug3_save_not_atomic_data_loss_on_crash(tmp_path, monkeypatch):
    """Crash giữa write phải không xóa sạch file inbox cũ."""
    from pathlib import Path
    from coworker.inbox import InboxStore

    path = tmp_path / "inbox.json"
    store = InboxStore(path)
    item = store.add_approval("s1", "Safe data")
    store.resolve(item.id, "allow")

    # Verify baseline: file tồn tại và có data
    assert path.exists() and "Safe data" in path.read_text()

    original_write_text = Path.write_text

    call_count = [0]

    def crashing_write(self, content, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1 and self == path:
            # Truncate (write_text làm điều này trước tiên) rồi crash
            self.write_bytes(b"")
            raise OSError("Simulated disk full mid-write")
        return original_write_text(self, content, **kwargs)

    monkeypatch.setattr(Path, "write_text", crashing_write)

    try:
        store.add_approval("s1", "New item triggers save")
    except OSError:
        pass  # expected

    monkeypatch.undo()

    # Reload: nếu save atomic, data cũ vẫn còn
    reloaded = InboxStore(path)
    items = reloaded.list(session_id="s1")

    assert len(items) >= 1, (
        "BUG 3 CONFIRMED: File bị truncate, tất cả data mất sau crash mid-write.\n"
        "Fix: write to <path>.tmp rồi os.replace(tmp, path)"
    )
    assert any(i.title == "Safe data" for i in items), (
        "BUG 3 CONFIRMED: Item 'Safe data' mất sau crash. Non-atomic write đã xóa nó."
    )


# ---------------------------------------------------------------------------
# BUG 4: LocalExecutor._bg_counter không thread-safe
#
# Source run_background() (shell.py):
#
#   def run_background(self, command):
#       self._bg_counter += 1          # <-- not atomic
#       task_id = f"bg-{self._bg_counter}"
#       self._bg_tasks[task_id] = task # <-- overwrite nếu duplicate ID
#
# Python GIL không đảm bảo i += 1 là atomic nếu có nhiều luồng.
# Concurrent calls có thể đọc cùng giá trị counter → cùng task_id
# → task sau overwrite task trước trong _bg_tasks → silent data loss.
# ---------------------------------------------------------------------------

def test_bug4_bg_counter_not_thread_safe(tmp_path):
    """Concurrent run_background() không được tạo duplicate task IDs."""
    from coworker.tools.shell import LocalExecutor

    ex = LocalExecutor(cwd=tmp_path, default_timeout=5)
    task_ids = []
    id_lock = threading.Lock()
    errors = []

    def launch():
        try:
            result = ex.run_background("echo bg")
            tid = result.get("task_id")
            if tid:
                with id_lock:
                    task_ids.append(tid)
        except Exception as e:
            with id_lock:
                errors.append(str(e))

    # 20 threads cùng lúc để maximize race condition
    threads = [threading.Thread(target=launch) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ex.close()

    assert not errors, f"Exceptions: {errors}"
    duplicates = len(task_ids) - len(set(task_ids))
    assert duplicates == 0, (
        f"BUG 4 CONFIRMED: {duplicates} duplicate task IDs trong {len(task_ids)} launches.\n"
        f"IDs: {sorted(task_ids)}\n"
        "Fix: dùng threading.Lock() bảo vệ _bg_counter += 1"
    )


# ---------------------------------------------------------------------------
# BUG 5: LocalExecutor._bg_tasks không pruned → memory leak
#
# Source: background_output() và background_kill() đọc/xóa task từ _bg_tasks
# nhưng KHÔNG bao giờ prune các task đã exit. Với session dài chạy hàng
# nghìn background tasks, _bg_tasks tích lũy vô hạn các dead _BackgroundTask
# objects + subprocess handles.
# ---------------------------------------------------------------------------

def test_bug5_bg_tasks_accumulate_dead_entries(tmp_path):
    """Dead background tasks phải được prune khỏi _bg_tasks sau khi exit."""
    from coworker.tools.shell import LocalExecutor

    ex = LocalExecutor(cwd=tmp_path, default_timeout=5)
    N = 20
    task_ids = []

    for _ in range(N):
        r = ex.run_background("echo done")
        task_ids.append(r["task_id"])

    # Chờ tất cả exit
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        all_done = all(
            task_ids[i] in ex._bg_tasks
            and ex._bg_tasks[task_ids[i]].proc.poll() is not None
            for i in range(N)
        )
        if all_done:
            break
        time.sleep(0.05)

    # Đọc output để trigger bất kỳ pruning logic nào (nếu có)
    for tid in task_ids:
        ex.background_output(tid)

    ex.close()

    # Đếm dead entries còn trong _bg_tasks
    dead = [
        tid for tid in task_ids
        if tid in ex._bg_tasks and ex._bg_tasks[tid].proc.poll() is not None
    ]

    # Nếu không có pruning: tất cả N tasks vẫn còn → bug confirmed
    assert len(dead) < N, (
        f"BUG 5 CONFIRMED: {len(dead)}/{N} exited tasks vẫn còn trong _bg_tasks.\n"
        "Fix: prune trong background_output() sau khi read tất cả output của task đã exit"
    )
