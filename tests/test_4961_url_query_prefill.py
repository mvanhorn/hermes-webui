"""Regression tests for #4961 URL query composer prefill behavior."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
SESSIONS_JS_PATH = REPO_ROOT / "static" / "sessions.js"
BOOT_JS_PATH = REPO_ROOT / "static" / "boot.js"
SESSIONS_JS = SESSIONS_JS_PATH.read_text(encoding="utf-8")
BOOT_JS = BOOT_JS_PATH.read_text(encoding="utf-8")
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node not on PATH")


def _run_node(source: str) -> str:
    result = subprocess.run(
        [NODE],
        input=source,
        cwd=str(REPO_ROOT),
        capture_output=True,
        encoding="utf-8",
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def _node_prelude() -> str:
    return f"""
const sessionsSrc = {SESSIONS_JS!r};
const bootSrc = {BOOT_JS!r};
function extractFunc(src, name) {{
  const re = new RegExp('(?:async\\\\s+)?function\\\\s+' + name + '\\\\s*\\\\(');
  const start = src.search(re);
  if (start < 0) throw new Error(name + ' not found');
  let i = src.indexOf('{{', start);
  let depth = 1; i++;
  while (depth > 0 && i < src.length) {{
    if (src[i] === '{{') depth++;
    else if (src[i] === '}}') depth--;
    i++;
  }}
  return src.slice(start, i);
}}
function evalSession(name) {{
  globalThis[name] = (0, eval)('(' + extractFunc(sessionsSrc, name) + ')');
}}
function evalBoot(name) {{
  globalThis[name] = (0, eval)('(' + extractFunc(bootSrc, name) + ')');
}}
"""


def test_prefill_intent_parses_q_prompt_and_send_flag():
    source = _node_prelude() + """
global.window = { location: { search: '?prompt=backup&q=hello%20world&send=YES' } };
evalSession('_composerPrefillIntentFromLocation');
const first = _composerPrefillIntentFromLocation();
window.location.search = '?prompt=from+prompt';
const second = _composerPrefillIntentFromLocation();
window.location.search = '?q=%20%20&send=1';
const third = _composerPrefillIntentFromLocation();
console.log(JSON.stringify({ first, second, third }));
"""
    payload = json.loads(_run_node(source))
    assert payload["first"] == {
        "hasParams": True,
        "hasText": True,
        "text": "hello world",
    }
    assert payload["second"] == {
        "hasParams": True,
        "hasText": True,
        "text": "from prompt",
    }
    assert payload["third"] == {
        "hasParams": True,
        "hasText": False,
        "text": "  ",
    }


def test_prefill_cleanup_removes_only_consumed_query_params():
    source = _node_prelude() + """
function applyUrl(rel) {
  const next = new URL(rel, 'https://example.test');
  window.location.href = next.href;
  window.location.pathname = next.pathname;
  window.location.search = next.search;
  window.location.hash = next.hash;
}
global.window = {
  location: {},
  history: {
    state: { from: 'test' },
    calls: [],
    replaceState(state, title, url) {
      this.calls.push({ state, title, url });
      this.state = state;
      applyUrl(url);
    }
  }
};
global.document = { baseURI: 'https://example.test/app/' };
applyUrl('/app/?q=hello&prompt=backup&send=1&session_id=target&keep=1#frag');
evalSession('_consumeComposerPrefillParamsFromLocation');
evalSession('_sessionUrlForSid');
_consumeComposerPrefillParamsFromLocation();
const cleaned = window.history.calls[0];
const promoted = _sessionUrlForSid('abc 123');
console.log(JSON.stringify({ cleaned, promoted }));
"""
    payload = json.loads(_run_node(source))
    assert payload["cleaned"]["url"] == "/app/?session_id=target&keep=1#frag"
    assert payload["cleaned"]["state"] == {"from": "test"}
    assert payload["promoted"] == "/app/session/abc%20123?keep=1#frag"


def test_root_prefill_keeps_saved_local_sidebar_only():
    source = _node_prelude() + """
evalBoot('_rootPrefillNeedsFreshComposer');
const result = {
  savedLocalWins: _rootPrefillNeedsFreshComposer(null, 'saved-local', { hasText: true }),
  explicitSessionWins: _rootPrefillNeedsFreshComposer('url-session', 'saved-local', { hasText: true }),
  blankPrefillIgnored: _rootPrefillNeedsFreshComposer(null, 'saved-local', { hasText: false })
};
console.log(JSON.stringify(result));
"""
    payload = json.loads(_run_node(source))
    assert payload == {
        "savedLocalWins": True,
        "explicitSessionWins": False,
        "blankPrefillIgnored": False,
    }
    prefill_guard = BOOT_JS.find("if(_rootPrefillNeedsFreshComposer(urlSession, savedLocal, prefillIntent)){")
    saved_guard = BOOT_JS.find("if(!urlSession&&savedLocal&&await _savedSessionShouldStaySidebarOnly(savedLocal)){")
    load_pos = BOOT_JS.find("await loadSession(saved);")
    assert prefill_guard >= 0
    assert saved_guard > prefill_guard
    assert load_pos > prefill_guard


def test_apply_prefill_updates_composer_without_autosend():
    source = _node_prelude() + """
(async () => {
  evalBoot('_applyComposerPrefillOnBoot');
  const counts = { autoResize: 0, updateSendBtn: 0, send: 0 };
  const msg = { value: '' };
  global.document = {
    getElementById(id) {
      return id === 'msg' ? msg : null;
    }
  };
  global.$ = (id) => document.getElementById(id);
  global.autoResize = () => { counts.autoResize++; };
  global.updateSendBtn = () => { counts.updateSendBtn++; };
  global.send = async () => { counts.send++; };
  await _applyComposerPrefillOnBoot({ hasText: true, text: 'hello world', autoSend: true });
  delete global.autoResize;
  await _applyComposerPrefillOnBoot({ hasText: true, text: 'second pass', autoSend: false });
  await _applyComposerPrefillOnBoot({ hasText: false, text: '   ', autoSend: true });
  console.log(JSON.stringify({ counts, value: msg.value }));
})().catch(err => {
  console.error(err);
  process.exit(1);
});
"""
    payload = json.loads(_run_node(source))
    assert payload["counts"] == {"autoResize": 1, "updateSendBtn": 1, "send": 0}
    assert payload["value"] == "second pass"


def test_explicit_session_url_keeps_target_session_with_prefill():
    source = _node_prelude() + """
global.window = {
  location: {
    pathname: '/session/url-target',
    search: '?q=hello%20world&send=1',
    hash: ''
  }
};
evalSession('_sessionIdFromLocation');
evalBoot('_rootPrefillNeedsFreshComposer');
console.log(JSON.stringify({
  urlSession: _sessionIdFromLocation(),
  bypassRootOverride: _rootPrefillNeedsFreshComposer(_sessionIdFromLocation(), 'saved-local', { hasText: true })
}));
"""
    payload = json.loads(_run_node(source))
    assert payload == {"urlSession": "url-target", "bypassRootOverride": False}
    query_source = _node_prelude() + """
global.window = {
  location: {
    pathname: '/',
    search: '?session_id=query-target&q=hello%20world&send=1',
    hash: ''
  }
};
evalSession('_sessionIdFromLocation');
evalBoot('_rootPrefillNeedsFreshComposer');
console.log(JSON.stringify({
  urlSession: _sessionIdFromLocation(),
  bypassRootOverride: _rootPrefillNeedsFreshComposer(_sessionIdFromLocation(), 'saved-local', { hasText: true })
}));
"""
    query_payload = json.loads(_run_node(query_source))
    assert query_payload == {"urlSession": "query-target", "bypassRootOverride": False}
    prefill_pos = BOOT_JS.find(
        "const prefillIntent=(typeof _composerPrefillIntentFromLocation==='function')?_composerPrefillIntentFromLocation():null;"
    )
    consume_pos = BOOT_JS.find("_consumeComposerPrefillParamsFromLocation();", prefill_pos)
    first_await_pos = BOOT_JS.find("const s=await api('/api/settings');", consume_pos)
    new_pos = BOOT_JS.find("await newSession(true);", consume_pos)
    load_pos = BOOT_JS.find("await loadSession(saved, {preserveActiveInput:true});", consume_pos)
    saved_pos = BOOT_JS.find("const saved=urlSession||savedLocal;")
    check_pos = BOOT_JS.find("await checkInflightOnBoot(saved);", load_pos)
    apply_pos = BOOT_JS.find("await _applyComposerPrefillOnBoot(prefillIntent);", check_pos)
    assert saved_pos >= 0
    assert prefill_pos >= 0
    assert consume_pos > prefill_pos
    assert 0 <= consume_pos < first_await_pos
    assert 0 <= consume_pos < new_pos
    assert 0 <= consume_pos < load_pos
    assert 0 <= check_pos < apply_pos
    zero_message_pos = BOOT_JS.find(
        "await renderSessionList();if(typeof startGatewaySSE==='function')startGatewaySSE();await _applyComposerPrefillOnBoot(prefillIntent);",
        load_pos,
    )
    assert zero_message_pos > load_pos
