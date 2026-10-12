"""Ensure third-party license notices are present in repo, wheel, and sdist."""
from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENDOR_LICENSES = ROOT / "src/automl/interfaces/web/static/vendor/licenses"

REQUIRED_ROOT_FILES = (
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_LICENSES.md",
)

REQUIRED_VENDOR_LICENSE_FILES = (
    "chartjs-MIT.txt",
    "lucide-ISC.txt",
    "tailwindcss-MIT.txt",
    "geist-OFL-1.1.txt",
    "ibm-plex-OFL-1.1.txt",
)

REQUIRED_VENDOR_ASSETS = (
    "chart.umd.min.js",
    "lucide.min.js",
    "VERSIONS.txt",
)


def test_repo_has_apache_license_and_third_party_inventory():
    for name in REQUIRED_ROOT_FILES:
        path = ROOT / name
        assert path.is_file(), f"Missing {name}"
        assert path.stat().st_size > 100, f"{name} looks empty"

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0" in license_text

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = "Apache-2.0"' in pyproject
    assert "THIRD_PARTY_LICENSES.md" in pyproject

    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "THIRD_PARTY_LICENSES.md" in notice

    inventory = (ROOT / "THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8")
    for needle in (
        "Chart.js",
        "Lucide",
        "Tailwind",
        "Geist",
        "IBM Plex Mono",
        "Apache License, Version 2.0",
        "does **not** change",
    ):
        assert needle in inventory


def test_vendor_license_texts_exist_beside_assets():
    for name in REQUIRED_VENDOR_LICENSE_FILES:
        path = VENDOR_LICENSES / name
        assert path.is_file(), f"Missing vendor license {name}"
        text = path.read_text(encoding="utf-8", errors="replace")
        assert len(text) > 80, f"{name} too short"

    versions = (ROOT / "src/automl/interfaces/web/static/vendor/VERSIONS.txt").read_text(
        encoding="utf-8"
    )
    assert "chart.js=4.4.7" in versions
    assert "lucide=0.469.0" in versions
    assert "tailwindcss=3.4.17" in versions


def test_vendor_license_texts_match_expected_families():
    chart = (VENDOR_LICENSES / "chartjs-MIT.txt").read_text(encoding="utf-8")
    assert "MIT License" in chart or "Permission is hereby granted" in chart

    lucide = (VENDOR_LICENSES / "lucide-ISC.txt").read_text(encoding="utf-8")
    assert "ISC" in lucide
    assert "Feather" in lucide or "MIT License" in lucide

    tw = (VENDOR_LICENSES / "tailwindcss-MIT.txt").read_text(encoding="utf-8")
    assert "MIT License" in tw
    assert "Tailwind Labs" in tw

    geist = (VENDOR_LICENSES / "geist-OFL-1.1.txt").read_text(encoding="utf-8")
    assert "SIL OPEN FONT LICENSE" in geist.upper() or "Open Font License" in geist

    plex = (VENDOR_LICENSES / "ibm-plex-OFL-1.1.txt").read_text(encoding="utf-8")
    assert "Open Font License" in plex or "SIL OPEN FONT LICENSE" in plex.upper()
    assert "Plex" in plex


@pytest.fixture(scope="module")
def built_artifacts(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Build sdist + wheel once for packaging assertions."""
    import subprocess
    import sys

    out = tmp_path_factory.mktemp("catml-license-dist")
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(out)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(out.glob("*.whl"))
    sdists = list(out.glob("*.tar.gz"))
    assert wheels, "wheel not produced"
    assert sdists, "sdist not produced"
    return wheels[0], sdists[0]


def test_wheel_includes_project_and_vendor_licenses(built_artifacts: tuple[Path, Path]):
    wheel_path, _ = built_artifacts
    with zipfile.ZipFile(wheel_path) as zf:
        names = set(zf.namelist())

    # Project license files in dist-info (setuptools license-files)
    dist_licenses = [n for n in names if "/licenses/" in n or n.endswith("LICENSE")]
    joined = "\n".join(sorted(names))
    assert "LICENSE" in joined
    assert "NOTICE" in joined
    assert "THIRD_PARTY_LICENSES.md" in joined

    # Vendored Workbench assets + license texts
    for asset in REQUIRED_VENDOR_ASSETS:
        assert any(n.endswith(f"vendor/{asset}") for n in names), f"wheel missing {asset}"

    for lic in REQUIRED_VENDOR_LICENSE_FILES:
        assert any(
            n.endswith(f"vendor/licenses/{lic}") for n in names
        ), f"wheel missing vendor/licenses/{lic}"

    # Fonts present => OFL texts must also be present (already checked above)
    assert any("/vendor/fonts/" in n and n.endswith(".woff2") for n in names)


def test_sdist_includes_project_and_vendor_licenses(built_artifacts: tuple[Path, Path]):
    _, sdist_path = built_artifacts
    with tarfile.open(sdist_path, "r:gz") as tf:
        names = set(tf.getnames())

    joined = "\n".join(sorted(names))
    assert "LICENSE" in joined
    assert "NOTICE" in joined
    assert "THIRD_PARTY_LICENSES.md" in joined

    for lic in REQUIRED_VENDOR_LICENSE_FILES:
        assert any(
            n.endswith(f"vendor/licenses/{lic}") for n in names
        ), f"sdist missing vendor/licenses/{lic}"


def test_catml_project_license_not_changed_to_mit():
    """Guardrail: compliance work must not flip the project license to MIT."""
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    # The historical MIT notice may appear in NOTICE, but LICENSE itself is Apache-2.0.
    assert "Apache License, Version 2.0" in license_text or "apache.org/licenses/LICENSE-2.0" in license_text
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = "Apache-2.0"' in pyproject
    assert 'license = "MIT"' not in pyproject
