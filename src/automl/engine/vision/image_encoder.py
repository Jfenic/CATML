from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Sequence
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from automl.domain.modalities.modality import Modality
from automl.domain.pipelines.graph import NodeType, PipelineNode


class ImageEncoderNode(BaseEstimator, TransformerMixin):
    """Execution engine node that transforms image collections into fixed-dimension numerical embeddings.

    Acts as the concrete execution adapter for an ENCODER stage in a PipelineGraph DAG,
    converting raw image file paths into a 2D float32 numpy array (N, D).
    """

    def __init__(
        self,
        node: PipelineNode | None = None,
        node_id: str = "image_encoder",
        output_dim: int | None = None,
        model_name: str = "deterministic",
        random_seed: int = 42,
        handle_missing: str = "raise",
        batch_size: int = 32,
        base_dir: str | Path | None = None,
        use_cache: bool = True,
        cache_dir: str | Path | None = None,
        allow_fallback: bool = False,
    ) -> None:
        """Initializes the image encoder node.

        Args:
            node: Optional domain PipelineNode to inherit configuration from.
            node_id: Unique identifier for this graph execution node.
            output_dim: Dimension of output embedding vectors (D). If None on neural backbones,
                        preserves native feature dimension without lossy truncation.
            model_name: Model/strategy name ('deterministic', 'hash', 'resnet18', etc.).
            random_seed: Reproducibility seed for deterministic projections.
            handle_missing: Policy for missing files: 'raise' or 'zero'.
            batch_size: Default chunk size for batch processing.
            base_dir: Optional base directory to prepend to relative image paths.
            use_cache: Whether to cache extracted embeddings to accelerate repeated runs.
            cache_dir: Optional directory for persistent disk caching of embeddings.
            allow_fallback: If False, requesting an unavailable neural backbone raises an error
                            instead of silently falling back to deterministic pseudorandom hashes.
        """
        self.node = node
        if node is not None:
            self.node_id = node.node_id
            params = dict(node.parameters)
            dim_val = params.get("output_dim", params.get("dim", output_dim))
            self.output_dim = int(dim_val) if dim_val is not None else None
            self.model_name = str(params.get("model_name", model_name))
            self.random_seed = int(params.get("random_seed", random_seed))
            self.handle_missing = str(params.get("handle_missing", handle_missing))
            self.batch_size = int(params.get("batch_size", batch_size))
            self.base_dir = params.get("base_dir", base_dir)
            self.use_cache = bool(params.get("use_cache", use_cache))
            self.cache_dir = params.get("cache_dir", cache_dir)
            self.allow_fallback = bool(params.get("allow_fallback", allow_fallback))
        else:
            self.node_id = node_id
            self.output_dim = output_dim
            self.model_name = model_name
            self.random_seed = random_seed
            self.handle_missing = handle_missing
            self.batch_size = batch_size
            self.base_dir = base_dir
            self.use_cache = use_cache
            self.cache_dir = cache_dir
            self.allow_fallback = allow_fallback

        if self.output_dim is not None and self.output_dim <= 0:
            raise ValueError(f"output_dim must be strictly positive, got {self.output_dim}")

        if self.model_name in ("deterministic", "hash") and self.output_dim is None:
            self.output_dim = 128

        self.input_modalities = (Modality.IMAGE,)
        self.output_modality = Modality.TABULAR
        self.is_fitted_ = False
        self._memory_cache: dict[str, np.ndarray] = {}

    @classmethod
    def from_pipeline_node(cls, node: PipelineNode) -> ImageEncoderNode:
        """Constructs an ImageEncoderNode directly from a domain PipelineNode."""
        return cls(node=node)

    def to_pipeline_node(self) -> PipelineNode:
        """Exports the configuration back to a pure domain PipelineNode."""
        return PipelineNode(
            node_id=self.node_id,
            node_type=NodeType.ENCODER,
            input_modalities=self.input_modalities,
            output_modality=self.output_modality,
            name=f"ImageEncoder({self.model_name})",
            parameters={
                "output_dim": self.output_dim,
                "model_name": self.model_name,
                "random_seed": self.random_seed,
                "handle_missing": self.handle_missing,
                "batch_size": self.batch_size,
                "base_dir": str(self.base_dir) if self.base_dir else None,
                "use_cache": self.use_cache,
                "cache_dir": str(self.cache_dir) if self.cache_dir else None,
                "allow_fallback": self.allow_fallback,
            },
        )

    def fit(self, X: Any, y: Any = None) -> ImageEncoderNode:
        """Scikit-learn compatible fit method."""
        self.is_fitted_ = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Scikit-learn compatible transform method returning 2D embeddings."""
        return self.encode(X)

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit and transform in a single call."""
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Returns generated embedding feature names for scikit-learn pipeline feature tracking."""
        prefix = f"{self.node_id}_" if self.node_id else "img_emb_"
        dim = self.output_dim if self.output_dim is not None else 512
        return np.array([f"{prefix}{i}" for i in range(dim)], dtype=object)

    def execute(self, inputs: Any) -> np.ndarray:
        """Executes node logic within a Pipeline DAG runner."""
        if isinstance(inputs, dict):
            data = inputs.get("data")
            if data is None:
                for k, v in inputs.items():
                    data = v
                    break
            return self.encode(data)
        return self.encode(inputs)

    def _resolve_path(self, item: str | Path | Any) -> Path:
        """Resolves an image path, prepending base_dir if the path is relative."""
        if (
            item is None
            or (isinstance(item, float) and np.isnan(item))
            or str(item).strip().lower() in ("none", "nan", "null", "")
        ):
            return Path("")
        p = Path(item)
        if not p.is_absolute() and self.base_dir is not None:
            p = Path(self.base_dir) / p
        return p

    def encode(self, image_paths: Sequence[str | Path | bytes] | Any) -> np.ndarray:
        """Encodes an iterable of image paths or image items into a 2D numpy array of shape (N, output_dim).

        Args:
            image_paths: Collection of image file paths (as strings, Paths, or bytes).

        Returns:
            2D numpy array of float32 embeddings with shape (N, output_dim).

        Raises:
            FileNotFoundError: If an image file does not exist and handle_missing == 'raise'.
            ValueError: If handle_missing policy is invalid.
        """
        # Convert pandas DataFrame/Series or numpy array to Python list
        if hasattr(image_paths, "iloc") and hasattr(image_paths, "ndim") and image_paths.ndim == 2:
            paths = image_paths.iloc[:, 0].tolist()
        elif hasattr(image_paths, "ndim") and image_paths.ndim == 2 and hasattr(image_paths, "flatten"):
            paths = image_paths.flatten().tolist()
        elif hasattr(image_paths, "tolist"):
            paths = image_paths.tolist()
        elif isinstance(image_paths, (list, tuple)):
            paths = list(image_paths)
        else:
            paths = [image_paths]

        clean_paths = []
        for p in paths:
            if isinstance(p, (list, tuple)) and len(p) == 1:
                clean_paths.append(p[0])
            else:
                clean_paths.append(p)
        paths = clean_paths

        n_samples = len(paths)
        if n_samples == 0:
            return np.empty((0, self.output_dim), dtype=np.float32)

        # Check model preference
        if self.model_name in ("deterministic", "hash"):
            embeddings = np.zeros((n_samples, self.output_dim), dtype=np.float32)
            for i, item in enumerate(paths):
                embeddings[i] = self._encode_single_with_cache(item)
            return embeddings

        # Neural vision backbones (resnet18, resnet50, mobilenet_v3_small, vit, timm, etc.)
        embeddings_dl = self._try_deep_learning_encode(paths)
        if embeddings_dl is not None:
            return embeddings_dl

        if not self.allow_fallback:
            raise RuntimeError(
                f"Vision model '{self.model_name}' failed to encode images and allow_fallback=False. "
                "Ensure PyTorch, torchvision, and Pillow are installed with pretrained weights, "
                "or specify model_name='deterministic' explicitly for tests."
            )

        # Fallback only when explicitly permitted
        embeddings = np.zeros((n_samples, self.output_dim), dtype=np.float32)
        for i, item in enumerate(paths):
            embeddings[i] = self._encode_single_with_cache(item)
        return embeddings

    def encode_batch(
        self,
        image_paths: Sequence[str | Path | bytes] | Any,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encodes images in batches to manage memory overhead during large inference tasks.

        Args:
            image_paths: Collection of image paths.
            batch_size: Optional batch size override.

        Returns:
            Concatenated 2D numpy array of embeddings.
        """
        bs = batch_size or self.batch_size
        if hasattr(image_paths, "tolist"):
            items = image_paths.tolist()
        else:
            items = list(image_paths)

        if not items:
            return np.empty((0, self.output_dim), dtype=np.float32)

        batches = [items[k : k + bs] for k in range(0, len(items), bs)]
        encoded_chunks = [self.encode(b) for b in batches]
        return np.vstack(encoded_chunks)

    def _get_cache_key(self, item: str | Path | bytes | Any) -> str:
        """Generates a unique cache key for an image item."""
        if isinstance(item, (bytes, bytearray)):
            digest = hashlib.sha256(item).hexdigest()[:16]
            return f"raw:{digest}:{self.output_dim}:{self.model_name}:{self.random_seed}"
        p = self._resolve_path(item)
        if str(p) == "":
            return f"missing:{self.output_dim}:{self.model_name}:{self.random_seed}"
        return f"file:{p.resolve()}:{self.output_dim}:{self.model_name}:{self.random_seed}"

    def _encode_single_with_cache(self, item: str | Path | bytes | Any) -> np.ndarray:
        """Retrieves or computes an embedding vector using the multi-level cache."""
        if not self.use_cache:
            return self._encode_single(item)

        cache_key = self._get_cache_key(item)

        # 1. Memory cache check
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        # 2. Disk cache check
        disk_path: Path | None = None
        if self.cache_dir is not None:
            cache_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
            disk_path = Path(self.cache_dir) / f"{cache_hash}.npy"
            if disk_path.is_file():
                try:
                    vec = np.load(disk_path).astype(np.float32)
                    self._memory_cache[cache_key] = vec
                    return vec
                except Exception:
                    pass

        # 3. Compute vector
        vector = self._encode_single(item)
        self._memory_cache[cache_key] = vector

        # Save to disk cache if configured
        if disk_path is not None:
            try:
                disk_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(disk_path, vector)
            except Exception:
                pass

        return vector

    def _encode_single(self, item: str | Path | bytes | Any) -> np.ndarray:
        """Computes a normalized embedding vector for a single image item."""
        if isinstance(item, (bytes, bytearray)):
            content_bytes = bytes(item)
        else:
            path = self._resolve_path(item)
            if str(path) == "" or not path.is_file():
                if self.handle_missing == "raise":
                    raise FileNotFoundError(f"Image path does not exist: {path}")
                elif self.handle_missing == "zero":
                    return np.zeros(self.output_dim, dtype=np.float32)
                else:
                    raise ValueError(f"Unknown handle_missing policy: '{self.handle_missing}'")

            try:
                with open(path, "rb") as f:
                    content_bytes = f.read()
            except Exception as e:
                if self.handle_missing == "raise":
                    raise IOError(f"Could not read image file '{path}': {e}") from e
                return np.zeros(self.output_dim, dtype=np.float32)

        # Generate deterministic pseudorandom embedding from image content hash
        hasher = hashlib.sha256()
        hasher.update(content_bytes)
        digest = hasher.digest()

        # Seed local generator with hash and node seed
        seed = (int.from_bytes(digest[:8], "big") ^ self.random_seed) % (2**32)
        rng = np.random.RandomState(seed)

        vector = rng.randn(self.output_dim).astype(np.float32)

        # Normalize to unit L2 norm
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm

        return vector

    def _try_deep_learning_encode(self, paths: list[Any]) -> np.ndarray | None:
        """Attempts to encode using torch/torchvision if available; falls back gracefully if not."""
        try:
            import torch  # type: ignore[import-untyped]
            import torchvision.models as tv_models  # type: ignore[import-untyped]
            import torchvision.transforms as T  # type: ignore[import-untyped]
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError as e:
            if not self.allow_fallback:
                raise RuntimeError(
                    f"Vision model '{self.model_name}' requires PyTorch, torchvision, and Pillow. "
                    "Install vision dependencies via `pip install 'catml[vision]'` or "
                    "specify model_name='deterministic' explicitly for testing."
                ) from e
            warnings.warn(
                "Vision framework PyTorch/torchvision or Pillow is not installed. "
                "Falling back transparently to lightweight deterministic embedding.",
                UserWarning,
                stacklevel=3,
            )
            return None

        try:
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            model_name = self.model_name.lower().replace("-", "_")
            if not hasattr(tv_models, model_name):
                if not self.allow_fallback:
                    raise ValueError(
                        f"Unsupported vision model backbone '{self.model_name}'. "
                        "Supported TorchVision backbones include: 'resnet18', 'resnet50', 'mobilenet_v3_small', etc."
                    )
                return None

            model_fn = getattr(tv_models, model_name)
            try:
                weights_attr = f"{model_name.title().replace('_', '')}_Weights"
                if hasattr(tv_models, weights_attr):
                    weights = getattr(tv_models, weights_attr).DEFAULT
                    model = model_fn(weights=weights)
                else:
                    model = model_fn(pretrained=True)
            except Exception as e:
                if not self.allow_fallback:
                    raise RuntimeError(
                        f"Failed to load pretrained weights for vision backbone '{self.model_name}': {e}. "
                        "CATML requires pretrained weights for semantic feature extraction and does "
                        "not use uninitialized random neural networks as a fallback."
                    ) from e
                warnings.warn(
                    f"Failed to load pretrained weights for '{self.model_name}' ({e}). Falling back to deterministic embedding.",
                    UserWarning,
                    stacklevel=3,
                )
                return None

            if hasattr(model, "fc"):
                model.fc = torch.nn.Identity()
            elif hasattr(model, "classifier"):
                model.classifier = torch.nn.Identity()
            elif hasattr(model, "heads"):
                model.heads = torch.nn.Identity()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()

            all_feats: list[np.ndarray] = []
            with torch.no_grad():
                for i in range(0, len(paths), self.batch_size):
                    batch_paths = paths[i : i + self.batch_size]
                    batch_tensors: list[torch.Tensor] = []
                    for p in batch_paths:
                        resolved = self._resolve_path(p)
                        if not resolved.is_file():
                            if self.handle_missing == "raise":
                                raise FileNotFoundError(f"Image path does not exist: {resolved}")
                            batch_tensors.append(torch.zeros(3, 224, 224))
                        else:
                            with Image.open(resolved) as img:
                                tensor = transform(img.convert("RGB"))
                                batch_tensors.append(tensor)

                    if not batch_tensors:
                        continue

                    stacked = torch.stack(batch_tensors).to(device)
                    feats = model(stacked)
                    if feats.dim() > 2:
                        feats = torch.flatten(feats, 1)

                    feats_np = feats.cpu().numpy().astype(np.float32)

                    if self.output_dim is None:
                        # Dynamically preserve native neural backbone dimensions (e.g. 512 for ResNet18)
                        self.output_dim = feats_np.shape[1]
                    elif feats_np.shape[1] != self.output_dim:
                        if feats_np.shape[1] > self.output_dim:
                            feats_np = feats_np[:, : self.output_dim]
                        else:
                            pad = np.zeros((feats_np.shape[0], self.output_dim - feats_np.shape[1]), dtype=np.float32)
                            feats_np = np.hstack([feats_np, pad])

                    norms = np.linalg.norm(feats_np, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    feats_np /= norms
                    all_feats.append(feats_np)

            if not all_feats:
                out_dim = self.output_dim if self.output_dim is not None else 512
                return np.empty((0, out_dim), dtype=np.float32)
            return np.vstack(all_feats)

        except Exception as e:
            if not self.allow_fallback:
                raise RuntimeError(
                    f"Vision model '{self.model_name}' inference failed: {e}. "
                    "Ensure valid image files and hardware configuration."
                ) from e
            warnings.warn(
                f"Vision model '{self.model_name}' inference encountered an issue ({e}). "
                f"Falling back transparently to lightweight deterministic embedding.",
                UserWarning,
                stacklevel=3,
            )
            return None
