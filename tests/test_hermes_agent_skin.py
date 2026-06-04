"""Hermes Agent skin: emerald/gold dashboard palette."""

from pathlib import Path


REPO = Path(__file__).parent.parent
CSS = (REPO / "static" / "style.css").read_text(encoding="utf-8")
BOOT_JS = (REPO / "static" / "boot.js").read_text(encoding="utf-8")
INDEX_HTML = (REPO / "static" / "index.html").read_text(encoding="utf-8")


def _css_block(selector):
    start = CSS.index(selector + "{")
    end = CSS.index("}", start)
    return CSS[start:end]


def test_hermes_agent_skin_present_in_picker_list():
    assert "{name:'Hermes Agent', value:'hermes-agent'" in BOOT_JS
    assert "'#0F1714','#16211C','#C89A5A'" in BOOT_JS


def test_hermes_agent_skin_in_early_init_allowlist():
    assert "'hermes-agent':1" in INDEX_HTML


def test_hermes_agent_dark_palette_uses_emerald_and_gold_tokens():
    block = _css_block(':root.dark[data-skin="hermes-agent"]')
    for token in (
        "--bg:#0F1714",
        "--sidebar:#121D18",
        "--surface:#16211C",
        "--border:#22342C",
        "--accent:#C89A5A",
        "--gold:#C89A5A",
        "--accent-hover:#D6AE74",
        "--accent-text:#E4C28D",
        "--success:#719A68",
    ):
        assert token in block, f"Hermes Agent palette token missing: {token}"


def test_hermes_agent_palette_omits_typography_and_shape_overrides():
    block = _css_block(':root.dark[data-skin="hermes-agent"]')
    for forbidden in (
        "--font-ui",
        "font-family:",
        "font-size:",
        "line-height:",
        "--radius-sm",
        "--radius-md",
        "--radius-card",
        "--radius-lg",
    ):
        assert forbidden not in block


def test_hermes_agent_skin_does_not_reskin_nous():
    nous_block = _css_block(':root.dark[data-skin="nous"]')
    assert "--bg:#0A0E14" in nous_block
    assert "--accent:#4682B4" in nous_block
    assert "--bg:#0F1714" not in nous_block
    assert "--accent:#C89A5A" not in nous_block
