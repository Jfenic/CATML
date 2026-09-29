# Feature Specification: V0.7 Multimodal Pipelines & Directed Acyclic Graph (DAG)

> Generalizing CATML to ingest, encode, and fuse multi-modal data (tabular + images) using a flexible DAG pipeline without altering existing tabular workflows.

---

## 1. Goal

Extend CATML's core capabilities beyond single-table datasets by:
1. Formalizing first-class domain concepts for **Data Modalities** (`Modality`, `DataSource`).
2. Transitioning from rigid linear execution to a typed **`PipelineGraph`** (Directed Acyclic Graph / DAG).
3. Implementing image ingestion and embedding extraction via an extensible **`ImageModalityPlugin`**.
4. Supporting **Feature Fusion** (early and late fusion) to evaluate tabular vs. image vs. multimodal experiments under the hypothesis-driven principle ("Propose ≠ Accept").

---

## 2. Business & Engineering Need

1. **Modern ML Competitions & Enterprise Problems:** Real-world datasets often mix structured attributes (metadata, pricing, user history) with visual evidence (product images, document scans, medical imaging).
2. **Backward Compatibility:** Existing tabular runs, benchmarks, and CLI commands must execute without modifications or performance degradation.
3. **Graph Type Safety:** Connecting incompatible pipeline stages (e.g. passing raw JPEG byte streams directly to Ridge regression instead of an encoder) must be validated and rejected early by `GraphValidator` before compute is allocated.
4. **Decoupled Collaboration:** The graph engine architecture and the image vision encoders must be strictly decoupled so two engineers can build them concurrently without merge conflicts.

---

## 3. Requirements

### 3.1 Domain Layer (`src/automl/domain/` — Pure Python)
- **`Modality` (Enum):**
  - Values: `TABULAR`, `IMAGE`, `TEXT`, `AUDIO`, `TIMESERIES`.
- **`DataSource` (Dataclass):**
  - Identifies a dataset source: `id: str`, `modality: Modality`, `path: str`, `metadata: dict[str, Any]`.
- **`PipelineNode` (Dataclass):**
  - Represents a processing stage: `node_id: str`, `node_type: str` ("source", "encoder", "fusion", "model"), `input_modalities: list[Modality]`, `output_modality: Modality`, `parameters: dict[str, Any]`.
- **`PipelineGraph` (Dataclass / Entity):**
  - Encapsulates nodes and directed edges `(source_node_id, target_node_id)`.
  - Provides topological traversal, node lookup, and dependency discovery.
- **`ModalityPluginPort` (Protocol in `src/automl/domain/ports.py`):**
  - Extends `PluginPort` for modal preprocessing and feature extraction.

### 3.2 Engine Layer (`src/automl/engine/`)
- **`GraphValidator` (`src/automl/engine/pipeline/graph_validator.py`):**
  - Detects cycles (cycle detection via DFS / topological sort).
  - Validates that connected edges have compatible modalities (`output_modality == input_modality`).
  - Verifies presence of at least one source node and one terminal estimator/model node.
- **`FeatureFusionNode` (`src/automl/engine/pipeline/fusion.py`):**
  - Horizontally concatenates tabular feature matrices with dense embedding arrays (`np.ndarray` / `pd.DataFrame`).
  - Ensures row-level alignment and handles missing modal inputs gracefully.
- **`ImageEncoderNode` (`src/automl/engine/vision/image_encoder.py`):**
  - Encodes raw image files or image path series into fixed-dimension numerical embedding vectors.
  - Supports mock/lightweight deterministic embeddings for unit testing without downloading multi-gigabyte neural network checkpoints.

### 3.3 Plugins Layer (`src/automl/plugins/`)
- **`ImageModalityPlugin` (`src/automl/plugins/modalities/image_plugin.py`):**
  - Declares `PluginCapability(supported_modalities=[Modality.IMAGE])`.
  - Implements image loading, dimension validation, and preprocessing hooks.

---

## 4. Constraints

1. **Domain Purity:** `src/automl/domain/` MUST NOT import `scikit-learn`, `PIL`, `torch`, `torchvision`, `numpy`, or `pandas`.
2. **Deterministic & Fast Testing:** The test suite must pass in `< 45s`. Test fixtures should use small synthetic in-memory images (e.g. $32 \times 32$ PNGs generated via Pillow).
3. **Graceful Fallback:** If heavy vision frameworks (PyTorch/TIMM) are not installed, the platform must remain fully functional using a lightweight or built-in encoder.

---

## 5. Acceptance Criteria

1. `PipelineGraph` builds and executes a multi-stage DAG with topological ordering.
2. `GraphValidator` raises `ValueError` on cyclic connections or mismatched input/output modalities.
3. `FeatureFusionNode` correctly aligns and concatenates tabular columns and embedding matrices.
4. `ImageModalityPlugin` extracts consistent embedding vectors from image paths.
5. All 95 existing tests pass without regressions, and new tests maintain global coverage $\ge 85\%$.
