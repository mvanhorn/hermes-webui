"""#7936: New Chat focuses the composer without a second awaited sidebar refresh.

``newSession()`` already schedules one un-awaited ``refreshSessionList``. The
``+`` button and Cmd/Ctrl+K handlers must not ``await renderSessionList()``
before focusing ``#msg``.
"""

from pathlib import Path


BOOT_JS = (Path(__file__).resolve().parents[1] / "static" / "boot.js").read_text(encoding="utf-8")
FRESH = "await newSession();closeMobileSidebar();$('msg').focus();"


def test_fresh_chat_does_not_await_a_second_sidebar_refresh():
    lines = [
        line.strip()
        for line in BOOT_JS.splitlines()
        if "await newSession();" in line and "closeMobileSidebar();" in line and "$('msg').focus();" in line
    ]
    assert lines == [FRESH, FRESH]
