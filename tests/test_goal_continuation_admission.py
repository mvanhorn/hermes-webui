"""Admission ownership for pending goal and background handoff markers.

Pending ``PENDING_GOAL_CONTINUATION`` and ``PENDING_BG_TASK_COMPLETIONS``
markers are in-memory handoff metadata. ``_start_chat_stream_for_session``
may consume them only for the attempt that passes the lock-held rejection
checks and session preparation. A 409, a preparation failure, or a rejected
regeneration leaves both markers in place. A failed ``Thread.start`` restores
only the markers that attempt consumed, and only while that stream still owns
the canonical session.

Refs #6885. Restart-durable intent, retries, and a continuation scheduler are
out of scope for this slice.
"""
from __future__ import annotations

import queue
import threading
import uuid
from contextlib import contextmanager

import pytest

import api.config as config
import api.routes as routes
from api.models import Session
from api.session_ops import plan_regeneration


class _CountingSet(set):
    def __init__(self, iterable=()):
        super().__init__(iterable)
        self.discard_calls = []

    def discard(self, element):
        self.discard_calls.append(element)
        super().discard(element)


class _Session:
    def __init__(self, session_id, **overrides):
        self.session_id = session_id
        self.title = "Existing chat"
        self.active_stream_id = None
        self.pending_user_message = None
        self.pending_attachments = []
        self.pending_started_at = None
        self.pending_user_source = None
        self.messages = [{"role": "user", "content": "old"}]
        self.context_messages = []
        self.workspace = "/tmp/goal-admission"
        self.model = "test-model"
        self.model_provider = None
        self.worktree_path = None
        self.profile = None
        self.session_source = "webui"
        self.post_compression_context_tokens_estimate = None
        self.saves = []
        for key, value in overrides.items():
            setattr(self, key, value)

    def save(self, *args, **kwargs):
        self.saves.append(self.active_stream_id)


class _TrackingThread(threading.Thread):
    started = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).started.append(self)


class _FailingThread:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def start(self):
        raise RuntimeError("thread launch failed")

    def join(self, timeout=None):
        return None


def _new_sid(label: str) -> str:
    return f"gcadm-{label}-{uuid.uuid4().hex[:10]}"


def _run_bounded(fn, timeout=5):
    """Run admission on a side thread so a non-reentrant lock deadlock fails the test."""
    box = {}

    def runner():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the test thread
            box["error"] = exc

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise AssertionError(
            "admission call did not finish; possible non-reentrant session-lock deadlock"
        )
    if "error" in box:
        raise box["error"]
    return box.get("value")


@contextmanager
def _handoff(session_id, *, goal=False, bg=False, counting=False):
    goal_set = _CountingSet() if counting else config.PENDING_GOAL_CONTINUATION
    bg_set = _CountingSet() if counting else config.PENDING_BG_TASK_COMPLETIONS
    previous_goal = routes.PENDING_GOAL_CONTINUATION
    previous_bg = routes.PENDING_BG_TASK_COMPLETIONS
    if counting:
        routes.PENDING_GOAL_CONTINUATION = goal_set
        routes.PENDING_BG_TASK_COMPLETIONS = bg_set
    if goal:
        routes.PENDING_GOAL_CONTINUATION.add(session_id)
    if bg:
        routes.PENDING_BG_TASK_COMPLETIONS.add(session_id)
    try:
        yield routes.PENDING_GOAL_CONTINUATION, routes.PENDING_BG_TASK_COMPLETIONS
    finally:
        routes.PENDING_GOAL_CONTINUATION.discard(session_id)
        routes.PENDING_BG_TASK_COMPLETIONS.discard(session_id)
        if counting:
            routes.PENDING_GOAL_CONTINUATION = previous_goal
            routes.PENDING_BG_TASK_COMPLETIONS = previous_bg
        else:
            config.PENDING_GOAL_CONTINUATION.discard(session_id)
            config.PENDING_BG_TASK_COMPLETIONS.discard(session_id)
        _release_session_runtime(session_id)


def _release_session_runtime(session_id: str) -> None:
    with config.STREAM_SESSION_OWNERS_LOCK:
        owned = [
            stream_id
            for stream_id, owner in list(config.STREAM_SESSION_OWNERS.items())
            if owner == session_id
        ]
        for stream_id in owned:
            config.STREAM_SESSION_OWNERS.pop(stream_id, None)
    with config.STREAMS_LOCK:
        for stream_id in owned:
            config.STREAMS.pop(stream_id, None)
    for stream_id in owned:
        config.STREAM_GOAL_RELATED.pop(stream_id, None)
    with config.SESSION_WRITEBACK_OWNERS_LOCK:
        config.SESSION_WRITEBACK_OWNERS.pop(session_id, None)
    with config.ACTIVE_RUNS_LOCK:
        for key, raw in list((config.ACTIVE_RUNS or {}).items()):
            if str((raw or {}).get("session_id") or "") == session_id:
                config.ACTIVE_RUNS.pop(key, None)
    try:
        from api import gateway_chat

        for stream_id in owned:
            gateway_chat._STREAM_RUN_LIFECYCLE.pop(stream_id, None)
            gateway_chat._STREAM_RUN_IDS.pop(stream_id, None)
    except Exception:
        pass


def _arm_busy_run(session_id: str, *, phase: str = "running", stream_id: str | None = None) -> str:
    stream_id = stream_id or f"busy-{uuid.uuid4().hex[:8]}"
    config.register_active_run(stream_id, session_id=session_id, phase=phase)
    return stream_id


def _install_admission_spies(monkeypatch, *, thread_cls=_TrackingThread):
    """Record journal/worker effects without running a provider or writing a journal file."""
    effects = {"prepare": [], "journal": [], "workers": []}
    _TrackingThread.started = []

    def prepare(session, **kwargs):
        effects["prepare"].append(kwargs.get("stream_id"))
        return routes_prepare(session, **kwargs)

    routes_prepare = routes._prepare_chat_start_session_for_stream

    def journal(*args, **kwargs):
        effects["journal"].append(args[1] if len(args) > 1 else kwargs.get("event"))
        return {"turn_id": "turn-admission"}

    def local_worker(*args, **kwargs):
        effects["workers"].append(("local", kwargs))

    def gateway_worker(*args, **kwargs):
        effects["workers"].append(("gateway", kwargs))

    monkeypatch.setattr(routes, "_prepare_chat_start_session_for_stream", prepare)
    monkeypatch.setattr(routes, "get_webui_session_save_mode", lambda: "deferred")
    monkeypatch.setattr(routes, "set_last_workspace", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(routes, "publish_session_list_changed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(routes, "create_stream_channel", lambda: queue.Queue())
    monkeypatch.setattr("api.turn_journal.append_turn_journal_event", journal)
    monkeypatch.setattr(routes, "_run_agent_streaming", local_worker)
    monkeypatch.setattr(routes, "_run_gateway_chat_streaming", gateway_worker)
    monkeypatch.setattr(routes.threading, "Thread", thread_cls)
    return effects


def _start(session, **kwargs):
    kwargs.setdefault("msg", "continue the goal")
    kwargs.setdefault("attachments", [])
    kwargs.setdefault("workspace", session.workspace)
    kwargs.setdefault("model", "test-model")
    kwargs.setdefault("model_provider", None)
    return routes._start_chat_stream_for_session(session, **kwargs)


def _assert_markers(session_id, *, goal, bg):
    assert (session_id in routes.PENDING_GOAL_CONTINUATION) is goal
    assert (session_id in routes.PENDING_BG_TASK_COMPLETIONS) is bg


@pytest.mark.parametrize("phase", ["running", "cancelling"])
def test_lifecycle_busy_rejection_preserves_both_markers(monkeypatch, phase):
    """A lifecycle-busy row with no active_stream_id must 409 without consuming markers.

    This is the premature-consumption bug: both markers were discarded before the
    session lock and the ACTIVE_RUNS guard.
    """
    sid = _new_sid(phase)
    session = _Session(sid)
    effects = _install_admission_spies(monkeypatch)
    busy_stream_id = _arm_busy_run(sid, phase=phase)
    try:
        with _handoff(sid, goal=True, bg=True):
            response = _start(session)
            assert response["_status"] == 409
            assert response["error"] == "session already has an active stream"
            assert response["active_stream_id"] == busy_stream_id
            _assert_markers(sid, goal=True, bg=True)
            assert effects["prepare"] == []
            assert effects["journal"] == []
            assert effects["workers"] == []
            assert session.active_stream_id is None
            assert session.pending_user_message is None
            assert session.saves == []
    finally:
        config.unregister_active_run(busy_stream_id)


def test_stale_stream_cleanup_then_busy_rejection_preserves_markers(monkeypatch):
    sid = _new_sid("stale-then-busy")
    session = _Session(sid, active_stream_id="stale-stream-id")
    effects = _install_admission_spies(monkeypatch)
    busy_stream_id = _arm_busy_run(sid, phase="cancelling")
    try:
        with _handoff(sid, goal=True, bg=True):
            response = _start(session)
            assert response["_status"] == 409
            assert response["active_stream_id"] == busy_stream_id
            _assert_markers(sid, goal=True, bg=True)
            assert effects["prepare"] == []
            assert effects["journal"] == []
            assert effects["workers"] == []
            assert session.active_stream_id is None
    finally:
        config.unregister_active_run(busy_stream_id)


def test_preparation_failure_preserves_markers(monkeypatch):
    sid = _new_sid("prepare-fail")
    session = _Session(sid)
    effects = _install_admission_spies(monkeypatch)

    def fail_prepare(session_obj, **kwargs):
        effects["prepare"].append(kwargs.get("stream_id"))
        session_obj.active_stream_id = kwargs.get("stream_id")
        session_obj.pending_user_message = kwargs.get("msg")
        raise RuntimeError("prepare failed")

    monkeypatch.setattr(routes, "_prepare_chat_start_session_for_stream", fail_prepare)
    with _handoff(sid, goal=True, bg=True):
        with pytest.raises(RuntimeError, match="prepare failed"):
            _start(session)
        _assert_markers(sid, goal=True, bg=True)
        assert effects["journal"] == []
        assert effects["workers"] == []
        assert len(effects["prepare"]) == 1


def test_thread_start_failure_restores_only_consumed_markers_while_owned(monkeypatch):
    sid = _new_sid("launch-fail")
    session = _Session(sid)
    _install_admission_spies(monkeypatch, thread_cls=_FailingThread)
    monkeypatch.setattr(routes, "get_session", lambda session_id, metadata_only=False: session)
    with _handoff(sid, goal=True, bg=True, counting=True) as (goal_set, bg_set):
        with pytest.raises(RuntimeError, match="thread launch failed"):
            _start(session, external_runtime_owned=False)
        assert goal_set.discard_calls == [sid]
        assert bg_set.discard_calls == [sid]
        _assert_markers(sid, goal=True, bg=True)
        assert session.active_stream_id is None
        assert session.pending_user_message is None


@pytest.mark.parametrize("gateway", [False, True])
def test_gateway_and_local_launch_failure_restores_owned_markers(monkeypatch, gateway):
    sid = _new_sid("launch-backend")
    session = _Session(sid)
    _install_admission_spies(monkeypatch, thread_cls=_FailingThread)
    monkeypatch.setattr(routes, "get_session", lambda session_id, metadata_only=False: session)
    with _handoff(sid, goal=True, bg=True):
        with pytest.raises(RuntimeError, match="thread launch failed"):
            _start(session, external_runtime_owned=gateway)
        _assert_markers(sid, goal=True, bg=True)
        assert session.active_stream_id is None


def test_launch_failure_does_not_resurrect_markers_for_deleted_session(monkeypatch):
    sid = _new_sid("launch-deleted")
    session = _Session(sid)
    _install_admission_spies(monkeypatch, thread_cls=_FailingThread)

    def deleted(_sid, metadata_only=False):
        raise KeyError(_sid)

    monkeypatch.setattr(routes, "get_session", deleted)
    with _handoff(sid, goal=True, bg=True, counting=True) as (goal_set, bg_set):
        with pytest.raises(RuntimeError, match="thread launch failed"):
            _start(session)
        assert goal_set.discard_calls == [sid]
        assert bg_set.discard_calls == [sid]
        _assert_markers(sid, goal=False, bg=False)
        assert session.pending_user_message == "continue the goal"
        assert session.saves == [session.active_stream_id]


def test_launch_failure_does_not_clear_successor_pending_state(monkeypatch):
    sid = _new_sid("launch-successor")
    session = _Session(sid)
    _install_admission_spies(monkeypatch, thread_cls=_FailingThread)

    def successor(_sid, metadata_only=False):
        session.active_stream_id = "successor-stream"
        session.pending_user_message = "successor prompt"
        session.pending_started_at = 50.0
        routes.PENDING_GOAL_CONTINUATION.add(_sid)
        return session

    monkeypatch.setattr(routes, "get_session", successor)
    with _handoff(sid, goal=True, bg=True, counting=True) as (goal_set, bg_set):
        with pytest.raises(RuntimeError, match="thread launch failed"):
            _start(session)
        assert goal_set.discard_calls == [sid]
        assert bg_set.discard_calls == [sid]
        _assert_markers(sid, goal=True, bg=False)
        assert session.active_stream_id == "successor-stream"
        assert session.pending_user_message == "successor prompt"
        assert session.pending_started_at == 50.0


def test_concurrent_callers_give_the_goal_marker_to_the_admitted_worker(monkeypatch):
    """Both callers pass the early check before either acquires the real session lock."""
    sid = _new_sid("race")
    session = _Session(sid)
    effects = _install_admission_spies(monkeypatch)
    barrier = threading.Barrier(2)
    release = threading.Event()
    passed = threading.Event()
    arrived = {"n": 0}
    arrived_lock = threading.Lock()
    acquired_locks = []
    real_get_lock = routes._get_session_agent_lock

    def wrapped(session_id):
        barrier.wait(timeout=5)
        with arrived_lock:
            arrived["n"] += 1
            if arrived["n"] == 2:
                passed.set()
        assert release.wait(timeout=5)
        lock = real_get_lock(session_id)
        acquired_locks.append(lock)
        return lock

    monkeypatch.setattr(routes, "_get_session_agent_lock", wrapped)
    results = []
    result_lock = threading.Lock()

    def call(message):
        try:
            value = _start(session, msg=message)
        except BaseException as exc:  # noqa: BLE001 - surfaced as the thread result
            value = exc
        with result_lock:
            results.append(value)

    workers = [
        threading.Thread(target=call, args=(f"goal turn {index}",))
        for index in (1, 2)
    ]
    with _handoff(sid, goal=True, bg=True):
        for worker in workers:
            worker.start()
        assert passed.wait(timeout=5), "both callers did not reach lock acquisition"
        try:
            _assert_markers(sid, goal=True, bg=True)
        finally:
            release.set()
        for worker in workers:
            worker.join(timeout=5)
            assert not worker.is_alive()
        for started in list(_TrackingThread.started):
            started.join(timeout=2)

        assert all(not isinstance(item, BaseException) for item in results), results
        admitted = [item for item in results if item.get("stream_id") and item.get("_status", 200) < 400 and "error" not in item]
        rejected = [item for item in results if item.get("_status") == 409]
        assert len(admitted) == 1, results
        assert len(rejected) == 1, results
        assert rejected[0]["error"] == "session already has an active stream"
        assert effects["workers"] == [("local", {"model_provider": None, "goal_related": True})]
        assert len(effects["prepare"]) == 1
        assert config.STREAM_GOAL_RELATED[admitted[0]["stream_id"]] is True
        assert session.pending_user_message in {"goal turn 1", "goal turn 2"}
        assert session.active_stream_id == admitted[0]["stream_id"]
        _assert_markers(sid, goal=False, bg=False)
        assert len(acquired_locks) == 2
        assert acquired_locks[0] is acquired_locks[1]


@pytest.mark.parametrize("gateway", [False, True])
def test_admitted_worker_receives_goal_classification(monkeypatch, gateway):
    sid = _new_sid("admit")
    session = _Session(sid)
    effects = _install_admission_spies(monkeypatch)
    with _handoff(sid, goal=True, bg=True, counting=True) as (goal_set, bg_set):
        response = _start(session, external_runtime_owned=gateway)
        for started in list(_TrackingThread.started):
            started.join(timeout=2)
        assert "error" not in response
        assert response["stream_id"]
        backend = "gateway" if gateway else "local"
        assert effects["workers"] == [(backend, {"model_provider": None, "goal_related": True})]
        assert goal_set.discard_calls == [sid]
        assert bg_set.discard_calls == [sid]
        _assert_markers(sid, goal=False, bg=False)
        assert config.STREAM_GOAL_RELATED[response["stream_id"]] is True
        assert session.active_stream_id == response["stream_id"]
        assert session.pending_user_message == "continue the goal"


def test_explicit_goal_related_keeps_pending_goal_marker_and_consumes_bg(monkeypatch):
    sid = _new_sid("explicit")
    session = _Session(sid)
    effects = _install_admission_spies(monkeypatch)
    with _handoff(sid, goal=True, bg=True, counting=True) as (goal_set, bg_set):
        response = _start(session, goal_related=True)
        for started in list(_TrackingThread.started):
            started.join(timeout=2)
        assert "error" not in response
        assert effects["workers"] == [("local", {"model_provider": None, "goal_related": True})]
        assert goal_set.discard_calls == []
        assert bg_set.discard_calls == [sid]
        _assert_markers(sid, goal=True, bg=False)


def test_explicit_goal_kickoff_uses_real_admission(monkeypatch, tmp_path):
    """/goal kickoff passes goal_related=True through the real admission function."""
    from api import goals as webui_goals

    sid = _new_sid("kickoff")
    session = _Session(sid, workspace=str(tmp_path), model="gpt-test", model_provider="openai")
    effects = _install_admission_spies(monkeypatch)

    class FakeState:
        goal = "ship the feature"
        status = "active"
        turns_used = 0
        max_turns = 20
        last_verdict = None
        last_reason = None
        paused_reason = None

    class FakeGoalManager:
        def __init__(self, session_id, default_max_turns=20):
            self.session_id = session_id
            self.state = None

        def set(self, goal):
            state = FakeState()
            state.goal = goal
            self.state = state
            return state

    monkeypatch.setattr(webui_goals, "GoalManager", FakeGoalManager)
    monkeypatch.setattr(routes, "get_session", lambda session_id, metadata_only=False: session)
    monkeypatch.setattr(routes, "resolve_trusted_workspace", lambda workspace, **_kwargs: workspace)
    monkeypatch.setattr(
        routes,
        "_resolve_compatible_session_model_state",
        lambda model, provider, **_kwargs: (model, provider, False),
    )
    monkeypatch.setattr(routes, "webui_gateway_chat_enabled", lambda _cfg: False)
    monkeypatch.setattr(routes, "get_config", lambda: {})
    monkeypatch.setattr(
        routes,
        "j",
        lambda handler, payload, status=200, **kwargs: {"status": status, "payload": payload},
    )
    with _handoff(sid, goal=True, bg=True):
        result = routes._handle_goal_command(
            object(),
            {
                "session_id": sid,
                "args": "ship the feature",
                "workspace": str(tmp_path),
                "model": "gpt-test",
                "model_provider": "openai",
            },
        )
        for started in list(_TrackingThread.started):
            started.join(timeout=2)
        assert result["status"] == 200
        assert result["payload"]["stream_id"]
        assert effects["workers"] == [("local", {"model_provider": "openai", "goal_related": True})]
        _assert_markers(sid, goal=True, bg=False)


def test_unmarked_chat_stays_unclassified(monkeypatch):
    sid = _new_sid("plain")
    session = _Session(sid)
    effects = _install_admission_spies(monkeypatch)
    with _handoff(sid):
        response = _start(session, msg="hello")
        for started in list(_TrackingThread.started):
            started.join(timeout=2)
        assert "error" not in response
        assert effects["workers"] == [("local", {"model_provider": None, "goal_related": False})]
        _assert_markers(sid, goal=False, bg=False)
        assert response["stream_id"] not in config.STREAM_GOAL_RELATED


def test_rejected_explicit_goal_start_preserves_both_markers(monkeypatch):
    sid = _new_sid("explicit-busy")
    session = _Session(sid)
    effects = _install_admission_spies(monkeypatch)
    busy_stream_id = _arm_busy_run(sid, phase="cancelling")
    try:
        with _handoff(sid, goal=True, bg=True):
            response = _start(session, goal_related=True)
            assert response["_status"] == 409
            _assert_markers(sid, goal=True, bg=True)
            assert effects["prepare"] == []
            assert effects["workers"] == []
    finally:
        config.unregister_active_run(busy_stream_id)


def _patch_server_turn(monkeypatch, session, *, gateway=False, legacy_journal=False):
    monkeypatch.setattr(routes, "get_session", lambda session_id, metadata_only=False: session)
    monkeypatch.setattr(
        routes,
        "_resolve_chat_workspace_with_recovery",
        lambda _session, _requested: session.workspace,
    )
    monkeypatch.setattr(
        routes,
        "_resolve_compatible_session_model_state",
        lambda model, provider, **_kwargs: (model or "test-model", provider, False),
    )
    monkeypatch.setattr(routes, "_agent_runtime_barrier_response", lambda **_kwargs: None)
    monkeypatch.setattr(routes, "webui_gateway_chat_enabled", lambda _cfg: gateway)
    if legacy_journal:
        monkeypatch.setenv("HERMES_WEBUI_RUNTIME_ADAPTER", "legacy-journal")
    else:
        monkeypatch.delenv("HERMES_WEBUI_RUNTIME_ADAPTER", raising=False)


@pytest.mark.parametrize("legacy_journal", [False, True])
def test_start_session_turn_busy_preserves_markers(monkeypatch, legacy_journal):
    sid = _new_sid("server-busy")
    session = _Session(sid)
    effects = _install_admission_spies(monkeypatch)
    _patch_server_turn(monkeypatch, session, legacy_journal=legacy_journal)
    busy_stream_id = _arm_busy_run(sid)
    adapter_calls = {"n": 0}
    from api import runtime_adapter

    real_build = runtime_adapter.build_runtime_adapter

    def track_build(**kwargs):
        adapter_calls["n"] += 1
        return real_build(**kwargs)

    monkeypatch.setattr(runtime_adapter, "build_runtime_adapter", track_build)
    try:
        with _handoff(sid, goal=True, bg=True):
            response = routes.start_session_turn(sid, "server continuation")
            assert response["_status"] == 409
            assert response["active_stream_id"] == busy_stream_id
            _assert_markers(sid, goal=True, bg=True)
            assert effects["prepare"] == []
            assert effects["journal"] == []
            assert effects["workers"] == []
            assert adapter_calls["n"] == (1 if legacy_journal else 0)
    finally:
        config.unregister_active_run(busy_stream_id)


@pytest.mark.parametrize("legacy_journal", [False, True])
@pytest.mark.parametrize("gateway", [False, True])
def test_start_session_turn_admits_goal_marker_through_real_delegate(
    monkeypatch, legacy_journal, gateway
):
    sid = _new_sid("server-admit")
    session = _Session(sid)
    effects = _install_admission_spies(monkeypatch)
    _patch_server_turn(
        monkeypatch, session, gateway=gateway, legacy_journal=legacy_journal
    )
    with _handoff(sid, goal=True, bg=True):
        response = routes.start_session_turn(sid, "server continuation")
        for started in list(_TrackingThread.started):
            started.join(timeout=2)
        assert response.get("_status", 200) < 400
        assert "error" not in response
        backend = "gateway" if gateway else "local"
        assert effects["workers"] == [
            (backend, {"model_provider": None, "goal_related": True})
        ]
        _assert_markers(sid, goal=False, bg=False)
        assert session.pending_user_message == "server continuation"


def _regeneration_session(sid, workspace):
    rows = [
        {"role": "user", "content": "prompt", "id": "u1", "_source": "webui"},
        {"role": "assistant", "content": "answer", "id": "a1"},
    ]
    session = Session(
        session_id=sid,
        messages=[dict(row) for row in rows],
        context_messages=[dict(row) for row in rows],
        workspace=workspace,
        title="Existing chat",
    )
    session.model = "test-model"
    session.model_provider = None
    return session


def test_rejected_regeneration_preserves_both_markers(monkeypatch, tmp_path):
    sid = _new_sid("regen-reject")
    session = _regeneration_session(sid, str(tmp_path))
    effects = _install_admission_spies(monkeypatch)
    monkeypatch.setattr(Session, "save", lambda self, *args, **kwargs: None)
    with _handoff(sid, goal=True, bg=True, counting=True) as (goal_set, bg_set):
        response = _run_bounded(
            lambda: _start(
                session,
                regeneration=type("Turn", (), {"revision": "stale-revision"})(),
            )
        )
        assert response["_status"] == 409
        assert response["code"] == "stale_regeneration_revision"
        assert goal_set.discard_calls == []
        assert bg_set.discard_calls == []
        _assert_markers(sid, goal=True, bg=True)
        assert effects["prepare"] == []
        assert effects["workers"] == []


def test_regeneration_apply_rejection_preserves_markers(monkeypatch, tmp_path):
    sid = _new_sid("regen-apply")
    session = _regeneration_session(sid, str(tmp_path))
    plan = plan_regeneration(session)
    effects = _install_admission_spies(monkeypatch)
    monkeypatch.setattr(Session, "save", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(
        "api.session_ops.apply_regeneration_plan",
        lambda *args, **kwargs: (False, None),
    )
    with _handoff(sid, goal=True, bg=True):
        response = _run_bounded(lambda: _start(session, regeneration=plan.turn))
        assert response["_status"] == 409
        assert response["code"] == "stale_regeneration_revision"
        _assert_markers(sid, goal=True, bg=True)
        assert effects["workers"] == []


@pytest.mark.parametrize("gateway", [False, True])
def test_accepted_regeneration_consumes_markers_once_and_classifies_worker(
    monkeypatch, tmp_path, gateway
):
    sid = _new_sid("regen-ok")
    session = _regeneration_session(sid, str(tmp_path))
    plan = plan_regeneration(session)
    effects = _install_admission_spies(monkeypatch)
    monkeypatch.setattr(Session, "save", lambda self, *args, **kwargs: None)
    with _handoff(sid, goal=True, bg=True, counting=True) as (goal_set, bg_set):
        response = _run_bounded(
            lambda: _start(
                session,
                regeneration=plan.turn,
                external_runtime_owned=gateway,
            )
        )
        for started in list(_TrackingThread.started):
            started.join(timeout=2)
        assert "error" not in response
        backend = "gateway" if gateway else "local"
        assert len(effects["workers"]) == 1
        assert effects["workers"][0][0] == backend
        assert effects["workers"][0][1]["goal_related"] is True
        assert goal_set.discard_calls == [sid]
        assert bg_set.discard_calls == [sid]
        _assert_markers(sid, goal=False, bg=False)
        assert config.STREAM_GOAL_RELATED[response["stream_id"]] is True


def test_accepted_explicit_regeneration_does_not_consume_goal_marker(monkeypatch, tmp_path):
    sid = _new_sid("regen-explicit")
    session = _regeneration_session(sid, str(tmp_path))
    plan = plan_regeneration(session)
    effects = _install_admission_spies(monkeypatch)
    monkeypatch.setattr(Session, "save", lambda self, *args, **kwargs: None)
    with _handoff(sid, goal=True, bg=True, counting=True) as (goal_set, bg_set):
        response = _run_bounded(
            lambda: _start(session, regeneration=plan.turn, goal_related=True)
        )
        for started in list(_TrackingThread.started):
            started.join(timeout=2)
        assert "error" not in response
        assert effects["workers"][0][1]["goal_related"] is True
        assert goal_set.discard_calls == []
        assert bg_set.discard_calls == [sid]
        _assert_markers(sid, goal=True, bg=False)


def test_regeneration_prepare_failure_preserves_markers(monkeypatch, tmp_path):
    sid = _new_sid("regen-prepare")
    session = _regeneration_session(sid, str(tmp_path))
    plan = plan_regeneration(session)
    _install_admission_spies(monkeypatch)
    monkeypatch.setattr(Session, "save", lambda self, *args, **kwargs: None)

    def fail_prepare(*args, **kwargs):
        raise RuntimeError("prepare failed")

    monkeypatch.setattr(routes, "_prepare_chat_start_session_for_stream", fail_prepare)
    with _handoff(sid, goal=True, bg=True):
        with pytest.raises(RuntimeError, match="prepare failed"):
            _run_bounded(lambda: _start(session, regeneration=plan.turn))
        _assert_markers(sid, goal=True, bg=True)


def test_regeneration_thread_start_failure_preserves_markers(monkeypatch, tmp_path):
    sid = _new_sid("regen-thread")
    session = _regeneration_session(sid, str(tmp_path))
    plan = plan_regeneration(session)
    _install_admission_spies(monkeypatch, thread_cls=_FailingThread)
    monkeypatch.setattr(Session, "save", lambda self, *args, **kwargs: None)
    with _handoff(sid, goal=True, bg=True, counting=True) as (goal_set, bg_set):
        with pytest.raises(RuntimeError, match="thread launch failed"):
            _run_bounded(lambda: _start(session, regeneration=plan.turn))
        assert goal_set.discard_calls == []
        assert bg_set.discard_calls == []
        _assert_markers(sid, goal=True, bg=True)


def test_regeneration_post_acceptance_failure_restores_owned_markers(monkeypatch, tmp_path):
    sid = _new_sid("regen-post")
    session = _regeneration_session(sid, str(tmp_path))
    plan = plan_regeneration(session)
    _install_admission_spies(monkeypatch)
    monkeypatch.setattr(Session, "save", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(routes, "get_session", lambda session_id, metadata_only=False: session)
    monkeypatch.setattr(
        routes,
        "set_last_workspace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("workspace failed")),
    )
    with _handoff(sid, goal=True, bg=True, counting=True) as (goal_set, bg_set):
        with pytest.raises(RuntimeError, match="workspace failed") as raised:
            _run_bounded(lambda: _start(session, regeneration=plan.turn))
        assert raised.value._regeneration_accepted is True
        assert goal_set.discard_calls == [sid]
        assert bg_set.discard_calls == [sid]
        _assert_markers(sid, goal=True, bg=True)
        assert session.active_stream_id


def test_regeneration_post_acceptance_failure_does_not_resurrect_deleted_session(
    monkeypatch, tmp_path
):
    sid = _new_sid("regen-deleted")
    session = _regeneration_session(sid, str(tmp_path))
    plan = plan_regeneration(session)
    _install_admission_spies(monkeypatch)
    monkeypatch.setattr(Session, "save", lambda self, *args, **kwargs: None)

    def deleted(_sid, metadata_only=False):
        raise KeyError(_sid)

    monkeypatch.setattr(routes, "get_session", deleted)
    monkeypatch.setattr(
        routes,
        "set_last_workspace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("workspace failed")),
    )
    with _handoff(sid, goal=True, bg=True, counting=True) as (goal_set, bg_set):
        with pytest.raises(RuntimeError, match="workspace failed"):
            _run_bounded(lambda: _start(session, regeneration=plan.turn))
        assert goal_set.discard_calls == [sid]
        assert bg_set.discard_calls == [sid]
        _assert_markers(sid, goal=False, bg=False)


def test_regeneration_post_acceptance_failure_keeps_successor_markers(monkeypatch, tmp_path):
    sid = _new_sid("regen-successor")
    session = _regeneration_session(sid, str(tmp_path))
    plan = plan_regeneration(session)
    _install_admission_spies(monkeypatch)
    monkeypatch.setattr(Session, "save", lambda self, *args, **kwargs: None)

    def successor(_sid, metadata_only=False):
        session.active_stream_id = "successor-stream"
        session.pending_user_message = "successor prompt"
        routes.PENDING_BG_TASK_COMPLETIONS.add(_sid)
        return session

    monkeypatch.setattr(routes, "get_session", successor)
    monkeypatch.setattr(
        routes,
        "set_last_workspace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("workspace failed")),
    )
    with _handoff(sid, goal=True, bg=True, counting=True) as (goal_set, bg_set):
        with pytest.raises(RuntimeError, match="workspace failed"):
            _run_bounded(lambda: _start(session, regeneration=plan.turn))
        assert goal_set.discard_calls == [sid]
        assert bg_set.discard_calls == [sid]
        _assert_markers(sid, goal=False, bg=True)
        assert session.active_stream_id == "successor-stream"
        assert session.pending_user_message == "successor prompt"
