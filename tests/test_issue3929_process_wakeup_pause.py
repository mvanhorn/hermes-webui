"""Regression coverage for #3929-D process-wakeup credential exhaustion pause."""

from __future__ import annotations

from pathlib import Path
import queue
import sys
import types
from unittest import mock

import pytest

import api.config as config
import api.models as models
import api.routes as routes
import api.streaming as streaming
from api.models import PROCESS_WAKEUP_PAUSE_ERROR, Session


@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", session_dir / "_index.json")
    models.SESSIONS.clear()
    yield
    models.SESSIONS.clear()


@pytest.fixture(autouse=True)
def _isolate_stream_state():
    config.STREAMS.clear()
    config.CANCEL_FLAGS.clear()
    config.AGENT_INSTANCES.clear()
    config.STREAM_PARTIAL_TEXT.clear()
    config.STREAM_REASONING_TEXT.clear()
    config.STREAM_LIVE_TOOL_CALLS.clear()
    config.STREAM_GOAL_RELATED.clear()
    config.PENDING_BG_TASK_COMPLETIONS.clear()
    with config.ACTIVE_RUNS_LOCK:
        config.ACTIVE_RUNS.clear()
    yield
    config.STREAMS.clear()
    config.CANCEL_FLAGS.clear()
    config.AGENT_INSTANCES.clear()
    config.STREAM_PARTIAL_TEXT.clear()
    config.STREAM_REASONING_TEXT.clear()
    config.STREAM_LIVE_TOOL_CALLS.clear()
    config.STREAM_GOAL_RELATED.clear()
    config.PENDING_BG_TASK_COMPLETIONS.clear()
    with config.ACTIVE_RUNS_LOCK:
        config.ACTIVE_RUNS.clear()


@pytest.fixture(autouse=True)
def _isolate_agent_locks():
    config.SESSION_AGENT_LOCKS.clear()
    yield
    config.SESSION_AGENT_LOCKS.clear()


@pytest.fixture(autouse=True)
def _mock_hermes_modules(monkeypatch):
    fake_runtime_module = types.ModuleType("hermes_cli.runtime_provider")
    fake_runtime_module.resolve_runtime_provider = lambda requested=None, **_kw: {
        "provider": requested or "test-provider",
        "api_key": "synthetic-key",
        "base_url": None,
    }
    fake_hermes_cli = types.ModuleType("hermes_cli")
    fake_hermes_cli.runtime_provider = fake_runtime_module
    fake_hermes_state = types.ModuleType("hermes_state")
    fake_hermes_state.SessionDB = mock.Mock(return_value=None)

    injected = {
        "hermes_cli": fake_hermes_cli,
        "hermes_cli.runtime_provider": fake_runtime_module,
        "hermes_state": fake_hermes_state,
    }
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in injected}
    sys.modules.update(injected)
    yield
    for name, previous in saved.items():
        if previous is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class _MockAgent:
    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id")
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.reasoning_callback = kwargs.get("reasoning_callback")
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.context_compressor = None
        self._last_error = None
        self.ephemeral_system_prompt = None

    def interrupt(self, _message):
        pass


class _CredentialPoolEmptyAgent(_MockAgent):
    def run_conversation(self, **_kwargs):
        raise RuntimeError("All 0 credential(s) exhausted for test-provider")


def _run_failing_process_wakeup(session: Session, tmp_path):
    stream_id = str(session.active_stream_id)
    fake_queue = queue.Queue()
    streaming.STREAMS[stream_id] = fake_queue
    config.STREAM_PARTIAL_TEXT[stream_id] = ""

    with mock.patch.object(streaming, "_get_ai_agent", return_value=_CredentialPoolEmptyAgent), \
         mock.patch.object(streaming, "resolve_model_provider", return_value=("test-model", "test-provider", None)), \
         mock.patch("api.config._resolve_cli_toolsets", return_value=[]):
        streaming._run_agent_streaming(
            session_id=session.session_id,
            msg_text=session.pending_user_message,
            model="test-model",
            model_provider="test-provider",
            workspace=str(tmp_path),
            stream_id=stream_id,
        )
    return [(item[0], item[1]) for item in list(fake_queue.queue)]


def test_credential_empty_process_wakeup_pauses_repeated_automatic_turns(tmp_path, monkeypatch):
    session = Session(
        session_id="wakeup_pause",
        title="Wakeup pause",
        workspace=str(tmp_path),
        model="test-model",
        model_provider="test-provider",
        messages=[{"role": "user", "content": "Earlier prompt", "timestamp": 1}],
        context_messages=[{"role": "user", "content": "Earlier prompt"}],
        active_stream_id="stream-wakeup-pause-1",
        pending_user_message="[IMPORTANT: Background process first completed.]",
        pending_started_at=1234.0,
        pending_user_source="process_wakeup",
    )
    session.save()
    models.SESSIONS[session.session_id] = session

    events = _run_failing_process_wakeup(session, tmp_path)
    saved = Session.load(session.session_id)
    assert saved is not None
    assert any(event == "apperror" and data["type"] == "credential_pool_empty" for event, data in events)
    assert sum(1 for message in saved.messages if message.get("_error")) == 1
    assert saved.process_wakeup_pause["paused"] is True
    assert saved.process_wakeup_pause["classification"] == "credential_pool_empty"
    assert saved.process_wakeup_pause["suppressed_count"] == 0
    context_before = list(saved.context_messages)
    messages_before = list(saved.messages)

    def _unexpected_start_run(*_args, **_kwargs):
        raise AssertionError("paused process_wakeup must not start another provider call")

    monkeypatch.setattr(routes, "_resolve_chat_workspace_with_recovery", lambda _s, _w: str(tmp_path))
    monkeypatch.setattr(routes, "_read_profile_model_config", lambda _s, _p: (None, None, {}))
    monkeypatch.setattr(
        routes,
        "_resolve_compatible_session_model_state",
        lambda *_args, **_kwargs: ("test-model", "test-provider", False),
    )
    monkeypatch.setattr(routes, "_start_run", _unexpected_start_run)

    response = routes.start_session_turn(
        session.session_id,
        "[IMPORTANT: Background process second completed.]",
        source="process_wakeup",
    )

    assert response["_status"] == 409
    assert response["error"] == PROCESS_WAKEUP_PAUSE_ERROR
    saved_after = Session.load(session.session_id)
    assert saved_after is not None
    assert saved_after.messages == messages_before
    assert saved_after.context_messages == context_before
    assert saved_after.process_wakeup_pause["suppressed_count"] == 1
    assert "last_suppressed_at" in saved_after.process_wakeup_pause


def test_process_wakeup_pause_resets_when_model_provider_lane_changes(tmp_path, monkeypatch):
    session = Session(
        session_id="wakeup_pause_reset",
        workspace=str(tmp_path),
        model="old-model",
        model_provider="old-provider",
        process_wakeup_pause={
            "version": 1,
            "paused": True,
            "source": "process_wakeup",
            "classification": "credential_pool_empty",
            "model": "old-model",
            "provider": "old-provider",
            "first_paused_at": 1.0,
            "last_error_at": 1.0,
            "visible_error_count": 1,
            "suppressed_count": 2,
        },
    )
    session.save()
    models.SESSIONS[session.session_id] = session

    captured = {}

    def _fake_start_run(s, **kwargs):
        captured["model"] = kwargs.get("model")
        captured["model_provider"] = kwargs.get("model_provider")
        return {"stream_id": "stream-reset", "session_id": s.session_id, "_status": 200}

    monkeypatch.setattr(routes, "_resolve_chat_workspace_with_recovery", lambda _s, _w: str(tmp_path))
    monkeypatch.setattr(routes, "_read_profile_model_config", lambda _s, _p: (None, None, {}))
    monkeypatch.setattr(
        routes,
        "_resolve_compatible_session_model_state",
        lambda *_args, **_kwargs: ("new-model", "new-provider", True),
    )
    monkeypatch.setattr(routes, "_start_run", _fake_start_run)

    response = routes.start_session_turn(
        session.session_id,
        "[IMPORTANT: Background process completed after provider change.]",
        source="process_wakeup",
    )

    assert response["_status"] == 200
    assert response["stream_id"] == "stream-reset"
    assert captured == {"model": "new-model", "model_provider": "new-provider"}
    saved = Session.load(session.session_id)
    assert saved is not None
    assert saved.process_wakeup_pause == {}


def test_success_path_clears_process_wakeup_pause_after_late_cancel_checks():
    src = Path(__file__).parent.parent.joinpath("api", "streaming.py").read_text(encoding="utf-8")
    session_save_idx = src.index('with _stream_writeback_stage(_writeback_timings, "session_save")')
    post_save_cancel_idx = src.index("if cancel_event.is_set():", session_save_idx)
    state_sync_idx = src.index('with _stream_writeback_stage(_writeback_timings, "state_sync")')
    final_cancel_idx = src.index("if cancel_event.is_set():", state_sync_idx)
    pause_clear_idx = src.index("clear_process_wakeup_pause(s, reason='run_completed')")
    done_payload_idx = src.index('with _stream_writeback_stage(_writeback_timings, "done_payload")')

    assert session_save_idx < post_save_cancel_idx < state_sync_idx
    assert state_sync_idx < final_cancel_idx < pause_clear_idx < done_payload_idx


def test_process_wakeup_pause_does_not_suppress_explicit_non_wakeup_turn(tmp_path, monkeypatch):
    session = Session(
        session_id="wakeup_pause_manual_recover",
        workspace=str(tmp_path),
        model="test-model",
        model_provider="test-provider",
        process_wakeup_pause={
            "version": 1,
            "paused": True,
            "source": "process_wakeup",
            "classification": "credential_pool_empty",
            "model": "test-model",
            "provider": "test-provider",
            "first_paused_at": 1.0,
            "last_error_at": 1.0,
            "visible_error_count": 1,
            "suppressed_count": 2,
        },
    )
    session.save()
    models.SESSIONS[session.session_id] = session

    captured = {}

    def _fake_start_run(s, **kwargs):
        captured["source"] = kwargs.get("source")
        captured["message"] = kwargs.get("msg")
        return {"stream_id": "stream-manual-recover", "session_id": s.session_id, "_status": 200}

    monkeypatch.setattr(routes, "_resolve_chat_workspace_with_recovery", lambda _s, _w: str(tmp_path))
    monkeypatch.setattr(routes, "_read_profile_model_config", lambda _s, _p: (None, None, {}))
    monkeypatch.setattr(
        routes,
        "_resolve_compatible_session_model_state",
        lambda *_args, **_kwargs: ("test-model", "test-provider", False),
    )
    monkeypatch.setattr(routes, "_start_run", _fake_start_run)

    response = routes.start_session_turn(
        session.session_id,
        "Explicit recovery attempt",
        source="manual_recover",
    )

    assert response["_status"] == 200
    assert response["stream_id"] == "stream-manual-recover"
    assert captured == {"source": "manual_recover", "message": "Explicit recovery attempt"}
    saved = Session.load(session.session_id)
    assert saved is not None
    assert saved.process_wakeup_pause["suppressed_count"] == 2
    assert "last_suppressed_at" not in saved.process_wakeup_pause
