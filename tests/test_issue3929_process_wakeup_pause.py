"""Regression coverage for #3929-D process-wakeup credential exhaustion pause."""

from __future__ import annotations

import json
from pathlib import Path
import queue
import hashlib
import sys
import types
from unittest import mock

import pytest

import api.config as config
import api.gateway_chat as gateway_chat
import api.models as models
import api.profiles as profiles
import api.providers as providers
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
def _default_live_credential_revalidation(monkeypatch):
    monkeypatch.setattr(
        routes,
        "provider_has_process_wakeup_recovery_credential",
        lambda *_args, **_kwargs: False,
    )


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


class _StaleCredentialPoolEmptyAgent(_MockAgent):
    def run_conversation(self, **_kwargs):
        session = models.SESSIONS[self.session_id]
        session.active_stream_id = "stream-newer-run"
        session.model = "newer-model"
        session.model_provider = "newer-provider"
        session.pending_user_source = "webui"
        session.save(touch_updated_at=False)
        raise RuntimeError("All 0 credential(s) exhausted for test-provider")


class _FakeCredentialPoolEntry:
    def __init__(self, payload):
        self._payload = dict(payload)
        for key, value in self._payload.items():
            setattr(self, key, value)
        self.source = self._payload.get("source", "manual")
        self.label = self._payload.get("label", "")
        self.key_source = self._payload.get("key_source", "")

    def to_dict(self):
        return dict(self._payload)


class _FakeCredentialPool:
    def __init__(self, entries):
        self._entries = list(entries)

    def entries(self):
        return list(self._entries)


def _install_fake_agent_credential_pool(monkeypatch, pool_data):
    fake_agent = types.ModuleType("agent")
    fake_agent.__path__ = []
    fake_pool_module = types.ModuleType("agent.credential_pool")

    def _load_pool(provider_id):
        entries = pool_data.get(provider_id, [])
        return _FakeCredentialPool(_FakeCredentialPoolEntry(entry) for entry in entries)

    fake_pool_module.load_pool = _load_pool
    monkeypatch.setitem(sys.modules, "agent", fake_agent)
    monkeypatch.setitem(sys.modules, "agent.credential_pool", fake_pool_module)


def _run_failing_process_wakeup(session: Session, tmp_path, *, stream_id=None):
    stream_id = str(stream_id or session.active_stream_id)
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


def _run_failing_process_wakeup_route(
    session: Session,
    tmp_path,
    *,
    route_model,
    route_provider=None,
    resolved_model,
    resolved_provider,
    resolved_base_url=None,
    custom_connection=(None, None),
    stream_id=None,
):
    stream_id = str(stream_id or session.active_stream_id)
    fake_queue = queue.Queue()
    streaming.STREAMS[stream_id] = fake_queue
    config.STREAM_PARTIAL_TEXT[stream_id] = ""

    with mock.patch.object(streaming, "_get_ai_agent", return_value=_CredentialPoolEmptyAgent), \
         mock.patch.object(
             streaming,
             "resolve_model_provider",
             return_value=(resolved_model, resolved_provider, resolved_base_url),
         ), \
         mock.patch.object(
             streaming,
             "resolve_custom_provider_connection",
             return_value=custom_connection,
         ), \
         mock.patch("api.config._resolve_cli_toolsets", return_value=[]):
        streaming._run_agent_streaming(
            session_id=session.session_id,
            msg_text=session.pending_user_message,
            model=route_model,
            model_provider=route_provider,
            workspace=str(tmp_path),
            stream_id=stream_id,
        )
    return [(item[0], item[1]) for item in list(fake_queue.queue)]


def _patch_process_wakeup_route(monkeypatch, tmp_path, *, model, provider):
    monkeypatch.setattr(routes, "_resolve_chat_workspace_with_recovery", lambda _s, _w: str(tmp_path))
    monkeypatch.setattr(routes, "_read_profile_model_config", lambda _s, _p: (None, None, {}))
    monkeypatch.setattr(
        routes,
        "_resolve_compatible_session_model_state",
        lambda *_args, **_kwargs: (model, provider, False),
    )


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
    assert saved.process_wakeup_pause["credential_state_fingerprint"]
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


def test_process_wakeup_pause_revalidates_when_credential_state_changes(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    auth_json = hermes_home / "auth.json"
    auth_json.write_text('{"credential_pool": {}}\n', encoding="utf-8")
    monkeypatch.setattr(models, "_get_profile_home", lambda _profile: hermes_home)
    session = Session(
        session_id="wakeup_pause_credential_refresh",
        workspace=str(tmp_path),
        model="test-model",
        model_provider="test-provider",
    )
    pause = models.record_process_wakeup_provider_unavailable_pause(
        session,
        classification="credential_pool_empty",
        model="test-model",
        provider="test-provider",
    )
    assert pause is not None
    paused_fingerprint = pause["credential_state_fingerprint"]
    session.save()
    models.SESSIONS[session.session_id] = session

    auth_json.write_text(
        '{"credential_pool": {"test-provider": [{"id": "refilled-token"}]}}\n',
        encoding="utf-8",
    )
    assert models.process_wakeup_credential_state_fingerprint(session) != paused_fingerprint

    captured = {}

    def _fake_start_run(s, **kwargs):
        captured["source"] = kwargs.get("source")
        captured["model"] = kwargs.get("model")
        captured["model_provider"] = kwargs.get("model_provider")
        return {"stream_id": "stream-credential-refresh", "session_id": s.session_id, "_status": 200}

    monkeypatch.setattr(routes, "_resolve_chat_workspace_with_recovery", lambda _s, _w: str(tmp_path))
    monkeypatch.setattr(routes, "_read_profile_model_config", lambda _s, _p: (None, None, {}))
    monkeypatch.setattr(
        routes,
        "_resolve_compatible_session_model_state",
        lambda *_args, **_kwargs: ("test-model", "test-provider", False),
    )
    monkeypatch.setattr(routes, "_start_run", _fake_start_run)
    monkeypatch.setattr(
        routes,
        "provider_has_process_wakeup_recovery_credential",
        lambda provider_id, *, refresh=False: provider_id == "test-provider",
    )

    response = routes.start_session_turn(
        session.session_id,
        "[IMPORTANT: Background process completed after credential refill.]",
        source="process_wakeup",
    )

    assert response["_status"] == 200
    assert response["stream_id"] == "stream-credential-refresh"
    assert captured == {
        "source": "process_wakeup",
        "model": "test-model",
        "model_provider": "test-provider",
    }
    saved = Session.load(session.session_id)
    assert saved is not None
    assert saved.process_wakeup_pause == {}


def test_process_wakeup_pause_keeps_changed_credential_state_until_provider_is_usable(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    auth_json = hermes_home / "auth.json"
    auth_json.write_text('{"credential_pool": {}}\n', encoding="utf-8")
    monkeypatch.setattr(models, "_get_profile_home", lambda _profile: hermes_home)
    session = Session(
        session_id="wakeup_pause_credential_changed_still_unusable",
        workspace=str(tmp_path),
        model="test-model",
        model_provider="test-provider",
    )
    pause = models.record_process_wakeup_provider_unavailable_pause(
        session,
        classification="credential_pool_empty",
        model="test-model",
        provider="test-provider",
    )
    assert pause is not None
    paused_fingerprint = pause["credential_state_fingerprint"]
    session.save()
    models.SESSIONS[session.session_id] = session

    auth_json.write_text(
        '{"credential_pool": {"test-provider": [{"id": "refilled-token"}]}}\n',
        encoding="utf-8",
    )
    changed_fingerprint = models.process_wakeup_credential_state_fingerprint(session)
    assert changed_fingerprint != paused_fingerprint

    def _unexpected_start_run(*_args, **_kwargs):
        raise AssertionError("changed credential metadata must not clear until the provider is usable")

    monkeypatch.setattr(routes, "_resolve_chat_workspace_with_recovery", lambda _s, _w: str(tmp_path))
    monkeypatch.setattr(routes, "_read_profile_model_config", lambda _s, _p: (None, None, {}))
    monkeypatch.setattr(
        routes,
        "_resolve_compatible_session_model_state",
        lambda *_args, **_kwargs: ("test-model", "test-provider", False),
    )
    monkeypatch.setattr(routes, "_start_run", _unexpected_start_run)
    monkeypatch.setattr(
        routes,
        "provider_has_process_wakeup_recovery_credential",
        lambda *_args, **_kwargs: False,
    )

    response = routes.start_session_turn(
        session.session_id,
        "[IMPORTANT: Background process completed after unusable credential metadata changed.]",
        source="process_wakeup",
    )

    assert response["_status"] == 409
    assert response["error"] == PROCESS_WAKEUP_PAUSE_ERROR
    saved = Session.load(session.session_id)
    assert saved is not None
    assert saved.process_wakeup_pause["suppressed_count"] == 1
    assert saved.process_wakeup_pause["credential_state_fingerprint"] == changed_fingerprint


def test_process_wakeup_pause_keeps_pause_when_config_key_matches_exhausted_pool_secret(tmp_path, monkeypatch):
    session = Session(
        session_id="wakeup_pause_config_key_matches_exhausted_pool",
        workspace=str(tmp_path),
        model="test-model",
        model_provider="test-provider",
    )
    pause = models.record_process_wakeup_provider_unavailable_pause(
        session,
        classification="credential_pool_empty",
        model="test-model",
        provider="test-provider",
    )
    assert pause is not None
    session.save()
    models.SESSIONS[session.session_id] = session
    exhausted_fingerprint = hashlib.sha256(b"same-key").hexdigest()
    monkeypatch.setattr(providers, "_get_provider_api_key", lambda _provider: "same-key")
    monkeypatch.setattr(
        providers,
        "_pool_entry_payloads",
        lambda _provider: [{
            "last_status": "dead",
            "secret_fingerprint": exhausted_fingerprint,
        }],
    )
    monkeypatch.setattr(
        routes,
        "provider_has_process_wakeup_recovery_credential",
        providers.provider_has_process_wakeup_recovery_credential,
    )

    def _unexpected_start_run(*_args, **_kwargs):
        raise AssertionError("same exhausted pool secret must not prove recovery")

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
        "[IMPORTANT: Background process completed with same exhausted config key.]",
        source="process_wakeup",
    )

    assert response["_status"] == 409
    assert response["error"] == PROCESS_WAKEUP_PAUSE_ERROR
    saved = Session.load(session.session_id)
    assert saved is not None
    assert saved.process_wakeup_pause["suppressed_count"] == 1


def test_process_wakeup_pause_clears_when_config_key_differs_from_exhausted_pool_secret(tmp_path, monkeypatch):
    session = Session(
        session_id="wakeup_pause_config_key_differs_from_exhausted_pool",
        workspace=str(tmp_path),
        model="test-model",
        model_provider="test-provider",
    )
    pause = models.record_process_wakeup_provider_unavailable_pause(
        session,
        classification="credential_pool_empty",
        model="test-model",
        provider="test-provider",
    )
    assert pause is not None
    session.save()
    models.SESSIONS[session.session_id] = session
    exhausted_fingerprint = hashlib.sha256(b"old-pool-key").hexdigest()
    monkeypatch.setattr(providers, "_get_provider_api_key", lambda _provider: "new-config-key")
    monkeypatch.setattr(
        providers,
        "_pool_entry_payloads",
        lambda _provider: [{
            "last_status": "dead",
            "secret_fingerprint": exhausted_fingerprint,
        }],
    )
    monkeypatch.setattr(
        routes,
        "provider_has_process_wakeup_recovery_credential",
        providers.provider_has_process_wakeup_recovery_credential,
    )

    captured = {}

    def _fake_start_run(s, **kwargs):
        captured["source"] = kwargs.get("source")
        captured["model"] = kwargs.get("model")
        captured["model_provider"] = kwargs.get("model_provider")
        return {"stream_id": "stream-config-key-recovered", "session_id": s.session_id, "_status": 200}

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
        "[IMPORTANT: Background process completed with new config key.]",
        source="process_wakeup",
    )

    assert response["_status"] == 200
    assert response["stream_id"] == "stream-config-key-recovered"
    assert captured == {
        "source": "process_wakeup",
        "model": "test-model",
        "model_provider": "test-provider",
    }
    saved = Session.load(session.session_id)
    assert saved is not None
    assert saved.process_wakeup_pause == {}


def test_process_wakeup_pause_survives_rotation_style_auth_rewrite(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    auth_json = hermes_home / "auth.json"
    auth_json.write_text(
        json.dumps(
            {
                "credential_pool": {
                    "other-provider": [
                        {
                            "id": "other-token",
                            "source": "oauth",
                            "auth_type": "oauth",
                            "access_token": "old-access",
                            "refresh_token": "old-refresh",
                            "expires_at": 1000,
                            "last_status": "ok",
                            "request_count": 1,
                            "updated_at": "old",
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(models, "_get_profile_home", lambda _profile: hermes_home)
    session = Session(
        session_id="wakeup_pause_auth_rotation",
        workspace=str(tmp_path),
        model="test-model",
        model_provider="test-provider",
    )
    pause = models.record_process_wakeup_provider_unavailable_pause(
        session,
        classification="credential_pool_empty",
        model="test-model",
        provider="test-provider",
    )
    assert pause is not None
    paused_fingerprint = pause["credential_state_fingerprint"]
    session.save()
    models.SESSIONS[session.session_id] = session

    auth_json.write_text(
        json.dumps(
            {
                "credential_pool": {
                    "other-provider": [
                        {
                            "id": "other-token",
                            "source": "oauth",
                            "auth_type": "oauth",
                            "access_token": "new-access",
                            "refresh_token": "new-refresh",
                            "expires_at": 2000,
                            "last_status": "ok",
                            "request_count": 2,
                            "updated_at": "new",
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    assert models.process_wakeup_credential_state_fingerprint(session) == paused_fingerprint

    def _unexpected_start_run(*_args, **_kwargs):
        raise AssertionError("token rotation must not clear a paused wakeup")

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
        "[IMPORTANT: Background process completed during token rotation.]",
        source="process_wakeup",
    )

    assert response["_status"] == 409
    assert response["error"] == PROCESS_WAKEUP_PAUSE_ERROR
    saved = Session.load(session.session_id)
    assert saved is not None
    assert saved.process_wakeup_pause["suppressed_count"] == 1
    assert saved.process_wakeup_pause["credential_state_fingerprint"] == paused_fingerprint


def test_process_wakeup_pause_revalidates_status_recovery_without_fingerprint_change(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    auth_json = hermes_home / "auth.json"
    exhausted_entry = {
        "id": "pooled-token",
        "label": "Test provider credential",
        "source": "manual",
        "auth_type": "api_key",
        "runtime_api_key": "sk-test-provider",
        "last_status": "exhausted",
        "last_status_at": "2026-07-07T00:00:00Z",
        "last_error_code": "429",
        "last_error_reset_at": "2999-01-01T00:00:00Z",
    }
    pool_data = {"test-provider": [dict(exhausted_entry)]}
    auth_json.write_text(json.dumps({"credential_pool": pool_data}), encoding="utf-8")
    _install_fake_agent_credential_pool(monkeypatch, pool_data)
    monkeypatch.setattr(models, "_get_profile_home", lambda _profile: hermes_home)
    monkeypatch.setattr(profiles, "get_active_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(
        routes,
        "provider_has_process_wakeup_recovery_credential",
        providers.provider_has_process_wakeup_recovery_credential,
    )
    session = Session(
        session_id="wakeup_pause_status_recovery",
        workspace=str(tmp_path),
        model="test-model",
        model_provider="test-provider",
    )
    pause = models.record_process_wakeup_provider_unavailable_pause(
        session,
        classification="credential_pool_empty",
        model="test-model",
        provider="test-provider",
    )
    assert pause is not None
    paused_fingerprint = pause["credential_state_fingerprint"]
    session.save()
    models.SESSIONS[session.session_id] = session

    recovered_entry = dict(exhausted_entry)
    recovered_entry["last_status"] = "ok"
    recovered_entry["last_status_at"] = "2026-07-07T00:01:00Z"
    recovered_entry["last_error_reset_at"] = None
    pool_data["test-provider"] = [recovered_entry]
    auth_json.write_text(json.dumps({"credential_pool": pool_data}), encoding="utf-8")
    assert models.process_wakeup_credential_state_fingerprint(session) == paused_fingerprint

    captured = {}

    def _fake_start_run(s, **kwargs):
        captured["source"] = kwargs.get("source")
        captured["model"] = kwargs.get("model")
        captured["model_provider"] = kwargs.get("model_provider")
        return {"stream_id": "stream-status-recovered", "session_id": s.session_id, "_status": 200}

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
        "[IMPORTANT: Background process completed after status recovery.]",
        source="process_wakeup",
    )

    assert response["_status"] == 200
    assert response["stream_id"] == "stream-status-recovered"
    assert captured == {
        "source": "process_wakeup",
        "model": "test-model",
        "model_provider": "test-provider",
    }
    saved = Session.load(session.session_id)
    assert saved is not None
    assert saved.process_wakeup_pause == {}


def test_process_wakeup_pause_revalidates_at_provider_model_with_canonical_provider(tmp_path, monkeypatch):
    session = Session(
        session_id="wakeup_pause_at_model_recovered",
        workspace=str(tmp_path),
        model="@test-provider:test-model",
        model_provider=None,
    )
    pause = models.record_process_wakeup_provider_unavailable_pause(
        session,
        classification="credential_pool_empty",
        model="test-model",
        provider="test-provider",
    )
    assert pause is not None
    session.save()
    models.SESSIONS[session.session_id] = session

    calls = []

    def _provider_has_process_wakeup_recovery_credential(provider_id, *, refresh=False):
        calls.append((provider_id, refresh))
        return provider_id == "test-provider"

    captured = {}

    def _fake_start_run(s, **kwargs):
        captured["source"] = kwargs.get("source")
        captured["model"] = kwargs.get("model")
        captured["model_provider"] = kwargs.get("model_provider")
        return {"stream_id": "stream-at-provider-recovered", "session_id": s.session_id, "_status": 200}

    _patch_process_wakeup_route(
        monkeypatch,
        tmp_path,
        model="@test-provider:test-model",
        provider=None,
    )
    monkeypatch.setattr(
        routes,
        "provider_has_process_wakeup_recovery_credential",
        _provider_has_process_wakeup_recovery_credential,
    )
    monkeypatch.setattr(routes, "_start_run", _fake_start_run)

    response = routes.start_session_turn(
        session.session_id,
        "[IMPORTANT: Background process completed after @provider recovery.]",
        source="process_wakeup",
    )

    assert response["_status"] == 200
    assert response["stream_id"] == "stream-at-provider-recovered"
    assert calls == [("test-provider", True)]
    assert captured == {
        "source": "process_wakeup",
        "model": "@test-provider:test-model",
        "model_provider": None,
    }
    saved = Session.load(session.session_id)
    assert saved is not None
    assert saved.process_wakeup_pause == {}


def test_process_wakeup_pause_revalidation_uses_session_profile_not_default(tmp_path, monkeypatch):
    base_home = tmp_path / "hermes-home"
    (base_home / "profiles" / "work").mkdir(parents=True)
    monkeypatch.setattr(profiles, "_DEFAULT_HERMES_HOME", base_home)
    monkeypatch.setattr(profiles, "_is_isolated_profile_mode", lambda: False)
    profiles.clear_request_profile()

    session = Session(
        session_id="wakeup_pause_named_profile_revalidation",
        workspace=str(tmp_path),
        model="test-model",
        model_provider="test-provider",
        profile="work",
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
            "suppressed_count": 0,
        },
    )
    session.process_wakeup_pause["credential_state_fingerprint"] = (
        models.process_wakeup_credential_state_fingerprint(session)
    )
    session.save()
    models.SESSIONS[session.session_id] = session

    calls = []

    def _provider_has_process_wakeup_recovery_credential(provider_id, *, refresh=False):
        active_profile = profiles.get_active_profile_name()
        calls.append((provider_id, refresh, active_profile))
        return active_profile == "default"

    def _unexpected_start_run(*_args, **_kwargs):
        raise AssertionError("named-profile wakeup must not clear from default-profile credentials")

    _patch_process_wakeup_route(
        monkeypatch,
        tmp_path,
        model="test-model",
        provider="test-provider",
    )
    monkeypatch.setattr(
        routes,
        "provider_has_process_wakeup_recovery_credential",
        _provider_has_process_wakeup_recovery_credential,
    )
    monkeypatch.setattr(routes, "_start_run", _unexpected_start_run)

    try:
        response = routes.start_session_turn(
            session.session_id,
            "[IMPORTANT: Background process completed while named profile is still exhausted.]",
            source="process_wakeup",
        )
    finally:
        profiles.clear_request_profile()

    assert response["_status"] == 409
    assert response["error"] == PROCESS_WAKEUP_PAUSE_ERROR
    assert calls == [("test-provider", True, "work")]
    saved = Session.load(session.session_id)
    assert saved is not None
    assert saved.process_wakeup_pause["suppressed_count"] == 1
    assert profiles.get_active_profile_name() == "default"


def test_process_wakeup_pause_suppresses_at_provider_model_session(tmp_path, monkeypatch):
    session = Session(
        session_id="wakeup_pause_at_model",
        workspace=str(tmp_path),
        model="@test-provider:test-model",
        model_provider=None,
    )
    pause = models.record_process_wakeup_provider_unavailable_pause(
        session,
        classification="credential_pool_empty",
        model="test-model",
        provider="test-provider",
    )
    assert pause is not None
    session.save()
    models.SESSIONS[session.session_id] = session

    def _unexpected_start_run(*_args, **_kwargs):
        raise AssertionError("@provider:model wakeup must be suppressed on the paused lane")

    monkeypatch.setattr(routes, "_resolve_chat_workspace_with_recovery", lambda _s, _w: str(tmp_path))
    monkeypatch.setattr(routes, "_read_profile_model_config", lambda _s, _p: (None, None, {}))
    monkeypatch.setattr(
        routes,
        "_resolve_compatible_session_model_state",
        lambda *_args, **_kwargs: ("@test-provider:test-model", None, False),
    )
    monkeypatch.setattr(routes, "_start_run", _unexpected_start_run)

    response = routes.start_session_turn(
        session.session_id,
        "[IMPORTANT: Background process completed for @provider:model lane.]",
        source="process_wakeup",
    )

    assert response["_status"] == 409
    assert response["error"] == PROCESS_WAKEUP_PAUSE_ERROR
    saved = Session.load(session.session_id)
    assert saved is not None
    assert saved.process_wakeup_pause["model"] == "test-model"
    assert saved.process_wakeup_pause["provider"] == "test-provider"
    assert saved.process_wakeup_pause["suppressed_count"] == 1


def test_process_wakeup_pause_suppresses_openrouter_tagged_model_session(tmp_path, monkeypatch):
    monkeypatch.setitem(
        config.cfg,
        "model",
        {"provider": "anthropic", "default": "claude-sonnet-4.6"},
    )
    assert models._process_wakeup_pause_key(
        "deepseek/deepseek-r1:free",
        "openrouter",
        "credential_pool_empty",
    ) == models._process_wakeup_pause_key(
        "deepseek/deepseek-r1:free",
        None,
        "credential_pool_empty",
    )
    session = Session(
        session_id="wakeup_pause_openrouter_tagged_model",
        workspace=str(tmp_path),
        model="deepseek/deepseek-r1:free",
        model_provider=None,
    )
    pause = models.record_process_wakeup_provider_unavailable_pause(
        session,
        classification="credential_pool_empty",
        model="deepseek/deepseek-r1:free",
        provider="openrouter",
    )
    assert pause is not None
    session.save()
    models.SESSIONS[session.session_id] = session

    def _unexpected_start_run(*_args, **_kwargs):
        raise AssertionError("OpenRouter tagged wakeup must be suppressed on the paused lane")

    monkeypatch.setattr(routes, "_resolve_chat_workspace_with_recovery", lambda _s, _w: str(tmp_path))
    monkeypatch.setattr(routes, "_read_profile_model_config", lambda _s, _p: (None, None, {}))
    monkeypatch.setattr(
        routes,
        "_resolve_compatible_session_model_state",
        lambda *_args, **_kwargs: ("deepseek/deepseek-r1:free", None, False),
    )
    monkeypatch.setattr(routes, "_start_run", _unexpected_start_run)

    response = routes.start_session_turn(
        session.session_id,
        "[IMPORTANT: Background process completed for OpenRouter tagged lane.]",
        source="process_wakeup",
    )

    assert response["_status"] == 409
    assert response["error"] == PROCESS_WAKEUP_PAUSE_ERROR
    saved = Session.load(session.session_id)
    assert saved is not None
    assert saved.process_wakeup_pause["model"] == "deepseek/deepseek-r1:free"
    assert saved.process_wakeup_pause["provider"] == "openrouter"
    assert saved.process_wakeup_pause["suppressed_count"] == 1


def test_process_wakeup_pause_suppresses_local_endpoint_custom_slug_session(tmp_path, monkeypatch):
    monkeypatch.setitem(
        config.cfg,
        "model",
        {
            "provider": "ollama",
            "default": "llama3.2",
            "base_url": "http://ollama.internal:11434/v1",
        },
    )
    endpoint_lane = models._process_wakeup_pause_key(
        "@custom:ollama.internal:11434:llama3.2",
        None,
        "credential_pool_empty",
    )
    assert models._process_wakeup_pause_key(
        "llama3.2",
        "ollama",
        "credential_pool_empty",
    ) == endpoint_lane
    session = Session(
        session_id="wakeup_pause_local_endpoint_slug",
        workspace=str(tmp_path),
        model="@custom:ollama.internal:11434:llama3.2",
        model_provider=None,
    )
    pause = models.record_process_wakeup_provider_unavailable_pause(
        session,
        classification="credential_pool_empty",
        model="llama3.2",
        provider="ollama",
    )
    assert pause is not None
    session.save()
    models.SESSIONS[session.session_id] = session

    def _unexpected_start_run(*_args, **_kwargs):
        raise AssertionError("local-endpoint wakeup must be suppressed on the resolved provider lane")

    monkeypatch.setattr(routes, "_resolve_chat_workspace_with_recovery", lambda _s, _w: str(tmp_path))
    monkeypatch.setattr(routes, "_read_profile_model_config", lambda _s, _p: (None, None, {}))
    monkeypatch.setattr(
        routes,
        "_resolve_compatible_session_model_state",
        lambda *_args, **_kwargs: ("@custom:ollama.internal:11434:llama3.2", None, False),
    )
    monkeypatch.setattr(routes, "_start_run", _unexpected_start_run)

    response = routes.start_session_turn(
        session.session_id,
        "[IMPORTANT: Background process completed for local endpoint lane.]",
        source="process_wakeup",
    )

    assert response["_status"] == 409
    assert response["error"] == PROCESS_WAKEUP_PAUSE_ERROR
    saved = Session.load(session.session_id)
    assert saved is not None
    assert saved.process_wakeup_pause["model"] == "llama3.2"
    assert saved.process_wakeup_pause["provider"] == endpoint_lane["provider"]
    assert saved.process_wakeup_pause["suppressed_count"] == 1


def test_process_wakeup_pause_suppresses_custom_provider_tagged_model_session(tmp_path, monkeypatch):
    session = Session(
        session_id="wakeup_pause_custom_tagged_model",
        workspace=str(tmp_path),
        model="@custom:proxy:model:tag",
        model_provider=None,
    )
    pause = models.record_process_wakeup_provider_unavailable_pause(
        session,
        classification="credential_pool_empty",
        model="model:tag",
        provider="custom:proxy",
    )
    assert pause is not None
    session.save()
    models.SESSIONS[session.session_id] = session

    def _unexpected_start_run(*_args, **_kwargs):
        raise AssertionError("custom-provider tagged wakeup must be suppressed on the paused lane")

    monkeypatch.setattr(routes, "_resolve_chat_workspace_with_recovery", lambda _s, _w: str(tmp_path))
    monkeypatch.setattr(routes, "_read_profile_model_config", lambda _s, _p: (None, None, {}))
    monkeypatch.setattr(
        routes,
        "_resolve_compatible_session_model_state",
        lambda *_args, **_kwargs: ("@custom:proxy:model:tag", None, False),
    )
    monkeypatch.setattr(routes, "_start_run", _unexpected_start_run)

    response = routes.start_session_turn(
        session.session_id,
        "[IMPORTANT: Background process completed for custom tagged lane.]",
        source="process_wakeup",
    )

    assert response["_status"] == 409
    assert response["error"] == PROCESS_WAKEUP_PAUSE_ERROR
    saved = Session.load(session.session_id)
    assert saved is not None
    assert saved.process_wakeup_pause["model"] == "model:tag"
    assert saved.process_wakeup_pause["provider"] == "custom:proxy"
    assert saved.process_wakeup_pause["suppressed_count"] == 1


def test_process_wakeup_pause_suppresses_custom_provider_single_segment_session(tmp_path, monkeypatch):
    assert models._process_wakeup_pause_key(
        "tag",
        "custom:proxy",
        "credential_pool_empty",
    ) == models._process_wakeup_pause_key(
        "@custom:proxy:tag",
        None,
        "credential_pool_empty",
    )
    session = Session(
        session_id="wakeup_pause_custom_single_segment",
        workspace=str(tmp_path),
        model="@custom:proxy:tag",
        model_provider=None,
    )
    pause = models.record_process_wakeup_provider_unavailable_pause(
        session,
        classification="credential_pool_empty",
        model="tag",
        provider="custom:proxy",
    )
    assert pause is not None
    session.save()
    models.SESSIONS[session.session_id] = session

    def _unexpected_start_run(*_args, **_kwargs):
        raise AssertionError("custom-provider shorthand wakeup must be suppressed on the paused lane")

    monkeypatch.setattr(routes, "_resolve_chat_workspace_with_recovery", lambda _s, _w: str(tmp_path))
    monkeypatch.setattr(routes, "_read_profile_model_config", lambda _s, _p: (None, None, {}))
    monkeypatch.setattr(
        routes,
        "_resolve_compatible_session_model_state",
        lambda *_args, **_kwargs: ("@custom:proxy:tag", None, False),
    )
    monkeypatch.setattr(routes, "_start_run", _unexpected_start_run)

    response = routes.start_session_turn(
        session.session_id,
        "[IMPORTANT: Background process completed for custom shorthand lane.]",
        source="process_wakeup",
    )

    assert response["_status"] == 409
    assert response["error"] == PROCESS_WAKEUP_PAUSE_ERROR
    saved = Session.load(session.session_id)
    assert saved is not None
    assert saved.process_wakeup_pause["model"] == "tag"
    assert saved.process_wakeup_pause["provider"] == "custom:proxy"
    assert saved.process_wakeup_pause["suppressed_count"] == 1


def test_process_wakeup_pause_records_route_lane_after_custom_runtime_rewrite(tmp_path, monkeypatch):
    session = Session(
        session_id="wakeup_pause_custom_runtime_rewrite",
        workspace=str(tmp_path),
        model="@custom:proxy:model:tag",
        model_provider=None,
        messages=[{"role": "user", "content": "Earlier prompt", "timestamp": 1}],
        context_messages=[{"role": "user", "content": "Earlier prompt"}],
        active_stream_id="stream-custom-runtime-rewrite",
        pending_user_message="[IMPORTANT: Background process first completed.]",
        pending_started_at=1234.0,
        pending_user_source="process_wakeup",
    )
    session.save()
    models.SESSIONS[session.session_id] = session

    events = _run_failing_process_wakeup_route(
        session,
        tmp_path,
        route_model="@custom:proxy:model:tag",
        route_provider=None,
        resolved_model="model:tag",
        resolved_provider="custom:proxy",
        custom_connection=(None, "http://proxy.example/v1"),
    )
    saved = Session.load(session.session_id)
    assert saved is not None
    assert any(event == "apperror" and data["type"] == "credential_pool_empty" for event, data in events)
    assert saved.process_wakeup_pause["model"] == "model:tag"
    assert saved.process_wakeup_pause["provider"] == "custom:proxy"

    def _unexpected_start_run(*_args, **_kwargs):
        raise AssertionError("post-rewrite custom-provider wakeup must be suppressed")

    _patch_process_wakeup_route(
        monkeypatch,
        tmp_path,
        model="@custom:proxy:model:tag",
        provider=None,
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
    assert saved_after.process_wakeup_pause["suppressed_count"] == 1


def test_process_wakeup_pause_records_unset_route_provider_before_runtime_backfill(tmp_path, monkeypatch):
    monkeypatch.setitem(config.cfg, "model", {"default": "claude-sonnet-test"})
    session = Session(
        session_id="wakeup_pause_unset_route_provider",
        workspace=str(tmp_path),
        model="claude-sonnet-test",
        model_provider=None,
        messages=[{"role": "user", "content": "Earlier prompt", "timestamp": 1}],
        context_messages=[{"role": "user", "content": "Earlier prompt"}],
        active_stream_id="stream-unset-route-provider",
        pending_user_message="[IMPORTANT: Background process first completed.]",
        pending_started_at=1234.0,
        pending_user_source="process_wakeup",
    )
    session.save()
    models.SESSIONS[session.session_id] = session

    events = _run_failing_process_wakeup_route(
        session,
        tmp_path,
        route_model="claude-sonnet-test",
        route_provider=None,
        resolved_model="claude-sonnet-test",
        resolved_provider=None,
    )
    saved = Session.load(session.session_id)
    assert saved is not None
    assert any(event == "apperror" and data["type"] == "credential_pool_empty" for event, data in events)
    assert saved.process_wakeup_pause["model"] == "claude-sonnet-test"
    assert saved.process_wakeup_pause["provider"] == ""

    def _unexpected_start_run(*_args, **_kwargs):
        raise AssertionError("runtime-provider backfill must not clear the paused route lane")

    _patch_process_wakeup_route(
        monkeypatch,
        tmp_path,
        model="claude-sonnet-test",
        provider=None,
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
    assert saved_after.process_wakeup_pause["suppressed_count"] == 1


def test_process_wakeup_pause_keeps_empty_provider_lane_after_fingerprint_change(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    auth_json = hermes_home / "auth.json"
    auth_json.write_text('{"credential_pool": {}}\n', encoding="utf-8")
    monkeypatch.setattr(models, "_get_profile_home", lambda _profile: hermes_home)
    monkeypatch.setitem(config.cfg, "model", {"default": "claude-sonnet-test"})
    session = Session(
        session_id="wakeup_pause_empty_provider_probe",
        workspace=str(tmp_path),
        model="claude-sonnet-test",
        model_provider=None,
    )
    pause = models.record_process_wakeup_provider_unavailable_pause(
        session,
        classification="credential_pool_empty",
        model="claude-sonnet-test",
        provider=None,
    )
    assert pause is not None
    assert pause["provider"] == ""
    paused_fingerprint = pause["credential_state_fingerprint"]
    session.save()
    models.SESSIONS[session.session_id] = session

    auth_json.write_text(
        '{"credential_pool": {"test-provider": [{"id": "refilled-token"}]}}\n',
        encoding="utf-8",
    )
    changed_fingerprint = models.process_wakeup_credential_state_fingerprint(session)
    assert changed_fingerprint != paused_fingerprint

    def _unexpected_start_run(*_args, **_kwargs):
        raise AssertionError("empty-provider lane must prove recovery before restarting")

    _patch_process_wakeup_route(
        monkeypatch,
        tmp_path,
        model="claude-sonnet-test",
        provider=None,
    )
    monkeypatch.setattr(
        routes,
        "provider_has_process_wakeup_recovery_credential",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(routes, "_start_run", _unexpected_start_run)

    response = routes.start_session_turn(
        session.session_id,
        "[IMPORTANT: Background process completed after empty-provider credential change.]",
        source="process_wakeup",
    )

    assert response["_status"] == 409
    assert response["error"] == PROCESS_WAKEUP_PAUSE_ERROR
    saved = Session.load(session.session_id)
    assert saved is not None
    assert saved.process_wakeup_pause["suppressed_count"] == 1
    assert saved.process_wakeup_pause["credential_state_fingerprint"] == changed_fingerprint


def test_gateway_cancel_during_completion_save_restores_process_wakeup_pause(tmp_path, monkeypatch):
    stream_id = "gateway-pause-save-race-stream"
    session_id = "gateway_pause_save_race"
    stream_queue = queue.Queue()
    config.STREAMS[stream_id] = stream_queue
    monkeypatch.setattr(gateway_chat, "RunJournalWriter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gateway_chat, "gateway_approval_unavailable_reason", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(config, "get_config", lambda: {"webui_gateway_base_url": "http://gateway.test"})

    class _GatewayResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def __iter__(self):
            payload = {"choices": [{"delta": {"content": "Gateway reply"}}]}
            return iter([
                ("data: " + json.dumps(payload) + "\n").encode("utf-8"),
                b"data: [DONE]\n",
            ])

    monkeypatch.setattr(
        gateway_chat.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _GatewayResponse(),
    )

    previous_pause = {
        "paused": True,
        "model": "claude-sonnet-test",
        "provider": "test-provider",
        "classification": "credential_pool_empty",
        "first_paused_at": 1.0,
        "last_visible_error_at": 1.0,
        "visible_error_count": 1,
        "suppressed_count": 0,
        "credential_state_fingerprint": "fingerprint-before",
    }
    previous_messages = [{"role": "user", "content": "before", "timestamp": 1.0}]
    session = Session(
        session_id=session_id,
        workspace=str(tmp_path),
        model="claude-sonnet-test",
        model_provider="test-provider",
        messages=list(previous_messages),
        context_messages=list(previous_messages),
        active_stream_id=stream_id,
        pending_user_message="wake up",
        pending_user_source="process_wakeup",
        process_wakeup_pause=dict(previous_pause),
    )
    session.save()
    models.SESSIONS[session_id] = session

    original_save = Session.save
    save_calls = {"count": 0, "cleared_pause": 0}

    def _save_and_cancel_after_success_clear(self, *args, **kwargs):
        save_calls["count"] += 1
        result = original_save(self, *args, **kwargs)
        if (
            getattr(self, "session_id", None) == session_id
            and not getattr(self, "process_wakeup_pause", None)
            and save_calls["cleared_pause"] == 0
        ):
            save_calls["cleared_pause"] += 1
            config.CANCEL_FLAGS[stream_id].set()
        return result

    monkeypatch.setattr(Session, "save", _save_and_cancel_after_success_clear)

    gateway_chat._run_gateway_chat_streaming(
        session_id,
        "wake up",
        "claude-sonnet-test",
        str(tmp_path),
        stream_id,
        model_provider="test-provider",
    )

    assert save_calls["cleared_pause"] == 1
    saved = Session.load(session_id)
    assert saved is not None
    assert saved.process_wakeup_pause == previous_pause
    assert saved.messages == previous_messages
    assert saved.context_messages == previous_messages
    queued_events = [item[0] for item in list(stream_queue.queue)]
    assert "cancel" in queued_events
    assert "done" not in queued_events


def test_stale_credential_empty_process_wakeup_still_records_pause(tmp_path):
    session = Session(
        session_id="wakeup_pause_stale",
        title="Wakeup pause stale",
        workspace=str(tmp_path),
        model="test-model",
        model_provider="test-provider",
        messages=[{"role": "user", "content": "Earlier prompt", "timestamp": 1}],
        context_messages=[{"role": "user", "content": "Earlier prompt"}],
        active_stream_id="stream-wakeup-pause-stale",
        pending_user_message="[IMPORTANT: Background process first completed.]",
        pending_started_at=1234.0,
        pending_user_source="process_wakeup",
    )
    session.save()
    models.SESSIONS[session.session_id] = session
    stream_id = str(session.active_stream_id)
    fake_queue = queue.Queue()
    streaming.STREAMS[stream_id] = fake_queue
    config.STREAM_PARTIAL_TEXT[stream_id] = ""

    with mock.patch.object(streaming, "_get_ai_agent", return_value=_StaleCredentialPoolEmptyAgent), \
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

    saved = Session.load(session.session_id)
    assert saved is not None
    assert saved.active_stream_id == "stream-newer-run"
    assert saved.model == "newer-model"
    assert saved.model_provider == "newer-provider"
    assert saved.pending_user_source == "webui"
    assert saved.process_wakeup_pause["paused"] is True
    assert saved.process_wakeup_pause["classification"] == "credential_pool_empty"
    assert saved.process_wakeup_pause["model"] == "test-model"
    assert saved.process_wakeup_pause["provider"] == "test-provider"
    assert saved.process_wakeup_pause["suppressed_count"] == 0
    assert not any(message.get("_error") for message in saved.messages)


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
    session_save_cancel_idx = src.index("if cancel_event.is_set():", session_save_idx)
    state_sync_idx = src.index('with _stream_writeback_stage(_writeback_timings, "state_sync")')
    final_cancel_idx = src.index("if cancel_event.is_set():", state_sync_idx)
    pause_snapshot_idx = src.index("_process_wakeup_pause_before_clear =", final_cancel_idx)
    pause_clear_idx = src.index("clear_process_wakeup_pause(s, reason='run_completed')")
    post_clear_cancel_idx = src.index("if cancel_event.is_set():", pause_clear_idx)
    post_clear_restore_idx = src.index(
        "s.process_wakeup_pause = dict(_process_wakeup_pause_before_clear)",
        post_clear_cancel_idx,
    )
    pause_save_idx = src.index('"process_wakeup_pause_clear_save"', post_clear_cancel_idx)
    post_save_cancel_idx = src.index("if cancel_event.is_set():", pause_save_idx)
    post_save_restore_idx = src.index(
        "s.process_wakeup_pause = dict(_process_wakeup_pause_before_clear)",
        post_save_cancel_idx,
    )
    done_payload_idx = src.index('with _stream_writeback_stage(_writeback_timings, "done_payload")')

    assert session_save_idx < session_save_cancel_idx < state_sync_idx
    assert state_sync_idx < final_cancel_idx < pause_snapshot_idx < pause_clear_idx
    assert pause_clear_idx < post_clear_cancel_idx < post_clear_restore_idx < pause_save_idx
    assert pause_save_idx < post_save_cancel_idx < post_save_restore_idx < done_payload_idx


def test_gateway_success_path_checks_cancel_before_clearing_process_wakeup_pause():
    src = Path(__file__).parent.parent.joinpath("api", "gateway_chat.py").read_text(encoding="utf-8")
    current_idx = src.index("if not _stream_writeback_is_current(s, stream_id):")
    early_cancel_idx = src.index("if cancel_event.is_set():", current_idx)
    pending_clear_idx = src.index("s.pending_user_source = None")
    pause_clear_idx = src.index('clear_process_wakeup_pause(s, reason="run_completed")')
    final_cancel_idx = src.rindex("if cancel_event.is_set():", current_idx, pause_clear_idx)
    save_idx = src.index("s.save()", pause_clear_idx)

    assert current_idx < early_cancel_idx < pending_clear_idx < final_cancel_idx < pause_clear_idx < save_idx


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
