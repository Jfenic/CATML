# AutoML Platform (V0.1)

Plataforma AutoML modular con arquitectura hexagonal, orientada a experimentos tabulares reproducibles.

## Requisitos

- Python 3.10+
- Dependencias: `numpy`, `pandas`, `scikit-learn`, `pyyaml`

## Instalación

```bash
cd CATML
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Ejemplo rápido

```bash
# Ejecutar un experimento con el dataset de ejemplo
automl run-demo

# Ejecutar tests
pytest
```

## Estructura

```text
src/automl/
├── domain/          # Entidades y contratos (Python puro)
├── application/     # Servicios y casos de uso
├── engine/          # Profiling, training, evaluation
├── plugins/         # Modelos (sklearn)
├── infrastructure/  # SQLite, storage
└── interfaces/      # CLI
```

## V0.1 — Alcance actual

- Registro de dataset y perfilado básico
- FeatureRegistry y ModelRegistry
- Creación y ejecución manual de Experiment/Trial
- Evaluación con holdout y cross-validation
- Persistencia en SQLite
- Dataset de ejemplo: `examples/data/customers_churn.csv`
