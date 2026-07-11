"""Regression: a stale sidebar unread dot must clear when a session is *visited*.

Salvage of #4946 (originally by @neaucode-bot), rebuilt fresh on master.

The yellow unread dot in the sidebar historically cleared only reliably when the
sidebar **row** was clicked. Opening/visiting a session through other paths — or
re-selecting the already-open session — could leave a stale dot, and a deferred
/api/sessions list poll landing across the async message-load gap could re-flag
the open session as unread.

The fix introduces `_acknowledgeSessionVisit()` (sync viewed count + polling
snapshot + repaint) and wires it into loadSession at three points:

  1. the same-session no-op guard (so re-selecting the open session clears a
     stale dot before returning),
  2. when the session metadata arrives, and
  3. again after the async message-load gap (so a deferred poll cannot leave a
     sticky dot).

Two invariants flagged in review are protected here and MUST NOT regress:

  (a) hidden/background completions must still be marked unread — the visit-ack
      does NOT loosen the focus gate on the completion paths, so a completion in
      a non-visible/non-focused tab is still flagged (concern a).
  (b) cleaning up a visited child's unread state must not strip a lineage
      PARENT's own unread dot — the visit repaints via
      renderSessionListFromCache(), which recomputes each row's aggregated
      unread authoritatively rather than doing ad-hoc DOM surgery (concern b).
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SESSIONS_JS = (ROOT / "static" / "sessions.js").read_text(encoding="utf-8")


def _load_session_block() -> str:
    start = SESSIONS_JS.index("async function loadSession(sid")
    end = SESSIONS_JS.index("function _resolveSessionModelForDisplaySoon", start)
    return SESSIONS_JS[start:end]


def _function_block(name: str, next_marker: str) -> str:
    start = SESSIONS_JS.index(f"function {name}")
    end = SESSIONS_JS.index(next_marker, start + 1)
    return SESSIONS_JS[start:end]


# ── Structural anchors ──────────────────────────────────────────────────────

def test_visit_ack_helpers_exist():
    assert "function _acknowledgeSessionVisit(sid, messageCount = 0, lastMessageAt = 0)" in SESSIONS_JS
    assert "function _syncSessionListSnapshotOnVisit(sid, messageCount, lastMessageAt)" in SESSIONS_JS
    assert "function _sessionVisitHasUnreadState(sid)" in SESSIONS_JS


def test_acknowledge_visit_syncs_viewed_snapshot_and_repaints():
    body = _function_block("_acknowledgeSessionVisit", "function _sessionVisitHasUnreadState")
    # Clears viewed count (which clears the stale completion-unread marker, #3020),
    # syncs the polling snapshot, and repaints the sidebar from cache.
    assert "_setSessionViewedCount(sid, messageCount);" in body
    assert "_syncSessionListSnapshotOnVisit(sid, messageCount, lastMessageAt);" in body
    assert "renderSessionListFromCache" in body


def test_load_session_acknowledges_visit_before_and_after_message_load():
    block = _load_session_block()
    # Metadata-arrival acknowledgment.
    first_ack = block.find("_acknowledgeSessionVisit(\n    S.session.session_id,")
    loading_clear = block.find("if (_isCurrentLoad()) _loadingSessionId = null;\n\n  // Re-acknowledge")
    second_ack = block.find("_acknowledgeSessionVisit(", loading_clear)

    assert first_ack != -1, "loadSession must acknowledge the visit when metadata arrives"
    assert loading_clear != -1, "loadSession must clear the in-flight marker before the final acknowledge"
    assert second_ack != -1 and first_ack < loading_clear < second_ack, (
        "loadSession must re-acknowledge after the async message-load gap so a "
        "deferred sidebar poll cannot leave a sticky unread dot"
    )


def test_same_session_reselect_clears_stale_unread():
    block = _load_session_block()
    guard = block.find("if(currentSid===sid && !forceReload && (!_loadingSessionId || _loadingSessionId===sid)){")
    unread_check = block.find("_sessionVisitHasUnreadState(sid)", guard)
    acknowledge = block.find("_acknowledgeSessionVisit(", unread_check)
    ret = block.find("return;", acknowledge)

    assert guard != -1, "same-session no-op guard must still exist"
    assert unread_check != -1 and guard < unread_check, (
        "re-selecting the already-open session must check for stale unread state"
    )
    assert acknowledge != -1 and unread_check < acknowledge < ret, (
        "re-selecting the already-open session must acknowledge the visit (clearing "
        "the stale dot) before returning"
    )


def test_completion_paths_keep_focus_gate_for_hidden_tab_completions():
    """Concern (a): the visit-ack must NOT loosen the completion paths' focus gate.

    A background completion in a hidden/unfocused tab must still be flagged
    unread, so the background + polling completion paths must keep using the
    focus-gated _isSessionActivelyViewedForList, not a focus-independent variant.
    """
    background = _function_block("_markSessionCompletionUnreadIfBackground", "function _clearSessionCompletionUnread")
    assert "_isSessionActivelyViewedForList(sid)" in background, (
        "background completion must keep the focus-gated read check so a hidden-tab "
        "completion is not prematurely marked read"
    )

    polling_start = SESSIONS_JS.index("function _markPollingCompletionUnreadTransitions(sessions)")
    polling_end = SESSIONS_JS.index("const staleRuntimeStateSids", polling_start)
    polling = SESSIONS_JS[polling_start:polling_end]
    assert "!_isSessionActivelyViewedForList(sid)" in polling, (
        "polling completion must keep the focus-gated read check so a hidden-tab "
        "completion is not prematurely marked read"
    )


# ── Functional behavior via node ────────────────────────────────────────────

def _extract(name: str) -> str:
    """Extract a top-level `function name(...) { ... }` definition by brace match."""
    marker = f"function {name}("
    start = SESSIONS_JS.index(marker)
    brace = SESSIONS_JS.index("{", start)
    depth = 0
    for i in range(brace, len(SESSIONS_JS)):
        ch = SESSIONS_JS[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return SESSIONS_JS[start:i + 1]
    raise AssertionError(f"could not brace-match {name}")


def _run_node(script: str) -> dict:
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def test_acknowledge_visit_clears_completion_unread_marker():
    """Visiting a session that carries an explicit completion-unread marker must
    clear it (so the yellow dot disappears) and repaint the sidebar."""
    ack = _extract("_acknowledgeSessionVisit")
    sync = _extract("_syncSessionListSnapshotOnVisit")
    set_viewed = _extract("_setSessionViewedCount")
    clear_unread = _extract("_clearSessionCompletionUnread")
    get_unread = _extract("_getSessionCompletionUnread")
    save_unread = _extract("_saveSessionCompletionUnread")
    get_counts = _extract("_getSessionViewedCounts")
    save_counts = _extract("_saveSessionViewedCounts")

    script = f"""
// Minimal localStorage shim.
const _store = {{}};
const localStorage = {{
  getItem: (k) => (k in _store ? _store[k] : null),
  setItem: (k, v) => {{ _store[k] = String(v); }},
}};
const SESSION_VIEWED_COUNTS_KEY = 'v';
const SESSION_COMPLETION_UNREAD_KEY = 'u';
let _sessionViewedCounts = null;
let _sessionCompletionUnread = null;
const _sessionListSnapshotById = new Map();
const _sessionStreamingById = new Map();
let S = {{ session: {{ session_id: 'open', message_count: 5 }} }};
let repaints = 0;
function renderSessionListFromCache() {{ repaints += 1; }}
function _forgetObservedStreamingSession() {{}}
{get_counts}
{save_counts}
{get_unread}
{save_unread}
{clear_unread}
{set_viewed}
{sync}
{ack}
// Seed a stale completion-unread marker for the open session.
_getSessionCompletionUnread()['open'] = {{message_count: 5, completed_at: 1}};
_saveSessionCompletionUnread();
const before = Object.prototype.hasOwnProperty.call(_getSessionCompletionUnread(), 'open');
_acknowledgeSessionVisit('open', 5, 10);
const after = Object.prototype.hasOwnProperty.call(_getSessionCompletionUnread(), 'open');
const snap = _sessionListSnapshotById.get('open');
console.log(JSON.stringify({{before, after, repaints, viewed: _getSessionViewedCounts()['open'], snap}}));
"""
    out = _run_node(script)
    assert out["before"] is True, "precondition: marker seeded"
    assert out["after"] is False, "visiting the session must clear the completion-unread marker"
    assert out["repaints"] >= 1, "visiting must repaint the sidebar from cache"
    assert out["viewed"] == 5, "viewed count must be synced to the current message count"
    assert out["snap"] == {"message_count": 5, "last_message_at": 10}, (
        "polling snapshot must be synced so a deferred list poll cannot re-flag the session"
    )


def test_visit_snapshot_prevents_deferred_poll_from_reflagging():
    """A deferred /api/sessions poll running just after a visit must NOT re-mark
    the open, unchanged session as a fresh background completion.

    This is the sticky-dot race: without the snapshot sync, the poll sees a
    session that "completed" relative to a stale/absent snapshot and re-flags it.
    """
    sync = _extract("_syncSessionListSnapshotOnVisit")
    set_viewed = _extract("_setSessionViewedCount")
    clear_unread = _extract("_clearSessionCompletionUnread")
    get_unread = _extract("_getSessionCompletionUnread")
    save_unread = _extract("_saveSessionCompletionUnread")
    get_counts = _extract("_getSessionViewedCounts")
    save_counts = _extract("_saveSessionViewedCounts")
    ack = _extract("_acknowledgeSessionVisit")
    transitions = _extract("_markPollingCompletionUnreadTransitions")
    effective = _extract("_isSessionEffectivelyStreaming")
    local = _extract("_isSessionLocallyStreaming")

    script = f"""
const _store = {{}};
const localStorage = {{
  getItem: (k) => (k in _store ? _store[k] : null),
  setItem: (k, v) => {{ _store[k] = String(v); }},
}};
const SESSION_VIEWED_COUNTS_KEY = 'v';
const SESSION_COMPLETION_UNREAD_KEY = 'u';
let _sessionViewedCounts = null;
let _sessionCompletionUnread = null;
const _sessionListSnapshotById = new Map();
const _sessionStreamingById = new Map();
// open + focused/visible so the focus gate treats it as actively viewed too.
let S = {{ session: {{ session_id: 'open', message_count: 5 }}, busy: false }};
let repaints = 0;
function renderSessionListFromCache() {{ repaints += 1; }}
function _forgetObservedStreamingSession(sid) {{}}
function _rememberObservedStreamingSession() {{}}
function _getSessionObservedStreaming() {{ return {{}}; }}
function _rememberSessionListSource() {{}}
function _hasPendingUserMessageSignal(s) {{ return !!(s && (s.pending_user_message || s.has_pending_user_message)); }}
function _markSessionCompletionUnread(sid, count) {{
  _getSessionCompletionUnread()[sid] = {{message_count: count, completed_at: 1}};
  _saveSessionCompletionUnread();
}}
const document = {{ visibilityState: 'visible', hasFocus: () => true }};
let _loadingSessionId = null;
const _allSessionsScope = null;
const _sessionListSourceById = new Map();
{get_counts}
{save_counts}
{get_unread}
{save_unread}
{clear_unread}
{set_viewed}
{local}
{effective}
{sync}
{ack}
function _isSessionActivelyViewedForList(sid) {{
  if (!sid || !S.session || S.session.session_id !== sid) return false;
  if (_loadingSessionId && _loadingSessionId !== sid) return false;
  if (document.visibilityState && document.visibilityState !== 'visible') return false;
  if (typeof document.hasFocus === 'function' && !document.hasFocus()) return false;
  return true;
}}
{transitions}
// Simulate: session was streaming, so a snapshot/streaming state exists.
_sessionStreamingById.set('open', true);
_sessionListSnapshotById.set('open', {{message_count: 4, last_message_at: 5}});
// User visits — acknowledge marks it read AND syncs the snapshot to current.
_acknowledgeSessionVisit('open', 5, 10);
// Now a deferred /api/sessions poll lands for the SAME (now idle) session.
_markPollingCompletionUnreadTransitions([
  {{session_id: 'open', is_streaming: false, active_stream_id: null, message_count: 5, last_message_at: 10, updated_at: 10}}
]);
const flagged = Object.prototype.hasOwnProperty.call(_getSessionCompletionUnread(), 'open');
console.log(JSON.stringify({{flagged}}));
"""
    out = _run_node(script)
    assert out["flagged"] is False, (
        "a deferred list poll after a visit must not re-flag the open, unchanged "
        "session as unread — the visit-ack synced snapshot + viewed count"
    )
