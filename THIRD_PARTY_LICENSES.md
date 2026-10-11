# Third-Party Licenses

This document inventories third-party software and fonts that CATML redistributes
or depends on. CATML itself remains licensed under the **Apache License, Version 2.0**
(see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). This file does **not** change
CATML’s license.

> **Commercial use:** The Workbench UI libraries and fonts listed below permit
> commercial use and redistribution when their notice/license conditions are met.
> Full license texts for vendored frontend assets ship with the Python package under
> `automl/interfaces/web/static/vendor/licenses/`.

---

## A. Vendored Workbench assets (redistributed in the wheel / sdist)

Pinned versions: [`src/automl/interfaces/web/static/vendor/VERSIONS.txt`](src/automl/interfaces/web/static/vendor/VERSIONS.txt).

| Component | Version | License | Source | Full text in package |
| --- | --- | --- | --- | --- |
| Chart.js | 4.4.7 | MIT | https://github.com/chartjs/Chart.js | `vendor/licenses/chartjs-MIT.txt` |
| Lucide (UMD) | 0.469.0 | ISC (+ MIT for Feather-derived icons) | https://github.com/lucide-icons/lucide | `vendor/licenses/lucide-ISC.txt` |
| Tailwind CSS | 3.4.17 (compiled offline to `css/tailwind.built.css`) | MIT | https://github.com/tailwindlabs/tailwindcss | `vendor/licenses/tailwindcss-MIT.txt` |
| Geist / Geist Mono | via `geist@1.3.1` woff2 | SIL OFL 1.1 | https://github.com/vercel/geist-font | `vendor/licenses/geist-OFL-1.1.txt` |
| IBM Plex Mono | via `@fontsource/ibm-plex-mono@5.2.5` woff2 | SIL OFL 1.1 (Reserved Font Name “Plex”) | https://github.com/IBM/plex | `vendor/licenses/ibm-plex-OFL-1.1.txt` |

### Font compliance notes (OFL-1.1)

- Fonts may be **bundled and sold with software**; they must **not** be sold as standalone font products.
- Each copy that redistributes the Font Software must include the copyright notice and OFL text (provided in `vendor/licenses/`).
- Modified versions must not use reserved names (**Geist**, **Plex**) without permission from the copyright holder.
- CATML redistributes unmodified woff2 files under their original names.

### Landing website fonts

The marketing site under `website/` also documents Geist under
[`website/public/FONT-LICENSES.txt`](website/public/FONT-LICENSES.txt). That tree is
**not** part of the PyPI wheel; Workbench compliance uses the vendor paths above.

---

## B. Core Python runtime dependencies (installed by pip; not vendored in the wheel)

These packages are declared in [`pyproject.toml`](pyproject.toml) and installed from
PyPI with their own license metadata. CATML does not re-bundle their source trees
inside the `catml` wheel. Typical SPDX / declared licenses (verify on upgrade):

| Package | Typical license |
| --- | --- |
| numpy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| scipy | BSD-3-Clause |
| optuna | MIT |
| PyYAML | MIT |
| joblib | BSD-3-Clause |

Optional extras (`models`, `vision`, `mcp`, `agents`, `parquet`) pull additional
packages (e.g. LightGBM, XGBoost, CatBoost, PyTorch, LangGraph). Those remain under
their upstream licenses when the user installs the corresponding extra.

---

## C. Build / vendor tooling (not redistributed in the runtime wheel)

| Tool | Role | License (typical) |
| --- | --- | --- |
| Node.js / npm | Used by `scripts/vendor_workbench_assets.sh` to fetch/build assets | Various |
| `@fontsource/ibm-plex-mono` | Source of Plex woff2 during vendor script | OFL-1.1 (font) |
| `geist` npm package | Source of Geist woff2 during vendor script | OFL-1.1 (font) |

---

## D. How to regenerate vendor license files

After updating pinned versions in `scripts/vendor_workbench_assets.sh`, refresh
license texts under `src/automl/interfaces/web/static/vendor/licenses/` from the
upstream tags/repos listed in section A, update `VERSIONS.txt`, and run:

```bash
.venv/bin/pytest tests/test_third_party_license_compliance.py -q
```

---

## E. Residual risks / out of scope

- **Optional heavy extras** (CUDA PyTorch, proprietary model weights): not audited as
  part of the default wheel; operators must review those stacks before redistribution.
- **Transitive npm build tools** used only at asset-build time are not shipped in the
  wheel; only the compiled Tailwind CSS and pinned JS/font binaries are.
- This inventory is **not legal advice**. For novel commercial terms, have counsel
  review the full texts in `vendor/licenses/` and upstream repositories.
