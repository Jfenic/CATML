"""Workbench UI chrome must ship offline (no public CDN script/font tags)."""
from __future__ import annotations

from pathlib import Path

STATIC = Path("src/automl/interfaces/web/static")

CDN_MARKERS = (
    "cdn.tailwindcss.com",
    "cdn.jsdelivr.net",
    "unpkg.com",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
)

REQUIRED_VENDOR = (
    "css/fonts.css",
    "css/tailwind.built.css",
    "vendor/chart.umd.min.js",
    "vendor/lucide.min.js",
    "vendor/VERSIONS.txt",
    "vendor/fonts/Geist-Regular.woff2",
    "vendor/fonts/GeistMono-Regular.woff2",
    "vendor/fonts/ibm-plex-mono-latin-400-normal.woff2",
)


def test_index_html_uses_local_vendor_assets_not_cdn() -> None:
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    for marker in CDN_MARKERS:
        assert marker not in index, f"index.html must not load {marker}"
    assert 'href="/static/css/fonts.css"' in index
    assert 'href="/static/css/tailwind.built.css"' in index
    assert 'src="/static/vendor/chart.umd.min.js"' in index
    assert 'src="/static/vendor/lucide.min.js"' in index
    assert "tailwind.config" not in index


def test_vendor_tree_and_built_tailwind_present() -> None:
    missing = [rel for rel in REQUIRED_VENDOR if not (STATIC / rel).is_file()]
    assert not missing, f"missing offline Workbench assets: {missing}"
    versions = (STATIC / "vendor/VERSIONS.txt").read_text(encoding="utf-8")
    assert "chart.js=" in versions
    assert "lucide=" in versions
    css = (STATIC / "css/tailwind.built.css").read_text(encoding="utf-8")
    assert "font-family:Geist" in css or "font-family: Geist" in css
    assert "createIcons" in (STATIC / "vendor/lucide.min.js").read_text(encoding="utf-8")
