# Publicar CATML en TestPyPI / PyPI (`0.9.0rc1`)

Checklist operativo para mantenedores. El workflow [`.github/workflows/release.yml`](../.github/workflows/release.yml) ya construye, verifica y publica vía **Trusted Publishing** (OIDC). No hay tokens de PyPI en el repositorio.

## Estado del código (automatizable)

Antes de publicar, `main` debe tener:

- [x] Stack tabular ship (#111): Workbench jobs, Audit-5 honesty, Clean Arch PR2/PR3
- [x] `license = "Apache-2.0"` + `setuptools>=77` en `pyproject.toml`
- [x] Cumplimiento third-party (`THIRD_PARTY_LICENSES.md` + textos en wheel) — `#112`
- [x] `python -m build` + `twine check` del wheel `0.9.0rc1`
- [x] Suite local reciente `pytest --cov-fail-under=85` (revalidar en CI del Release)

## Lo que solo pueden hacer los mantenedores (humano)

Acuerdo de [@Jfenic](https://github.com/Jfenic) y [@tvarmar](https://github.com/tvarmar).

### 1. GitHub Environments

Ya creados: `testpypi` y `pypi`, con required reviewers (Jfenic, tvarmar).

Comprobar en: https://github.com/Jfenic/catml/settings/environments

### 2. Trusted Publisher en TestPyPI

1. Cuenta en https://test.pypi.org (misma org/proyecto que usaréis en PyPI).
2. Crear proyecto **`catml`** (o “pending publisher” si aún no existe).
3. **Publishing → Trusted Publishers → Add**:
   - Owner: `Jfenic`
   - Repository: `catml`
   - Workflow: `release.yml`
   - Environment: `testpypi`

### 3. Trusted Publisher en PyPI

Igual en https://pypi.org para el proyecto `catml`, environment `pypi`, workflow `release.yml`.

### 4. Publicar rc1

**Opción A — manual (recomendada la primera vez):**

1. Actions → **Release** → *Run workflow*
2. Target: `testpypi`
3. Aprobar el environment `testpypi` cuando lo pida GitHub
4. Verificar: `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ catml==0.9.0rc1`
5. Smoke: `catml --help` y `python -c "from catml import AutoML; print('ok')"`

**Opción B — tag (producción):**

```bash
# En main, versión == 0.9.0-rc1 en src/automl/__init__.py
git tag -a v0.9.0-rc1 -m "CATML 0.9.0-rc1"
git push origin v0.9.0-rc1
```

El push del tag dispara build → verify → `publish-pypi` (environment `pypi` con aprobación).

> Normalización tag↔versión: `v0.9.0-rc1` ↔ `0.9.0-rc1` (el workflow compara sin puntuación).

### 5. Tras TestPyPI OK

1. Anunciar en README / GitHub Release notes (CHANGELOG sección `[0.9.0-rc1]`).
2. Publicar a PyPI (dispatch `pypi` o tag).
3. Verificar: `pip install catml==0.9.0rc1`

## Qué no hace falta para rc1

- Scaffold de Workbench Labs (Vision/NLP UI) — diferido ([ADR-011](decisions/011-workbench-labs-modular-frontend.md))
- Benchmarks AMLB públicos
- Migración a React

## Contacto

Si Trusted Publisher falla (OIDC / environment name mismatch), revisar que el nombre del environment en PyPI coincida exactamente con `testpypi` / `pypi` del workflow.
