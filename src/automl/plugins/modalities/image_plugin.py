from __future__ import annotations

import os
from pathlib import Path
import struct
from typing import Any, Sequence

from automl.domain.modalities.modality import DataSource, Modality
from automl.domain.plugins.plugin import PluginCapability, PluginType
from automl.domain.ports import ModalityPluginPort


SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".gif",
    ".tiff",
    ".tif",
)


def is_image_column(series: Any, sample_size: int = 200) -> bool:
    """Heuristic to determine if a pandas Series contains image file paths or image references
    rather than general categorical tokens, freeform text, or raw numeric data.
    """
    if not hasattr(series, "dropna"):
        return False

    try:
        import pandas as pd

        if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
            return False
    except ImportError:
        pass

    clean = series.dropna().astype(str)
    if len(clean) == 0:
        return False

    if len(clean) > sample_size:
        clean = clean.sample(sample_size, random_state=42)

    valid_count = 0
    for val in clean:
        v = str(val).strip().lower()
        if not v or v in ("none", "nan", "null"):
            continue
        if any(v.endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS):
            valid_count += 1
        elif os.path.isfile(val):
            if Path(val).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                valid_count += 1

    match_ratio = valid_count / len(clean)
    return match_ratio >= 0.50


def get_image_metadata(file_path: str | Path) -> dict[str, Any]:
    """Reads dimensions and metadata from image headers without external dependencies.

    Args:
        file_path: Path to the image file.

    Returns:
        Dictionary with image file metadata (path, name, format, size_bytes, width, height).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    size_bytes = p.stat().st_size
    ext = p.suffix.lower()

    # Attempt to load using PIL if available in the environment
    try:
        from PIL import Image  # type: ignore[import-untyped]

        with Image.open(p) as img:
            return {
                "path": str(p.resolve()),
                "file_name": p.name,
                "extension": ext,
                "size_bytes": size_bytes,
                "format": img.format.lower() if img.format else ext.lstrip("."),
                "width": img.width,
                "height": img.height,
                "mode": getattr(img, "mode", None),
            }
    except Exception:
        pass

    # Standard library fallback parser for common headers
    width: int | None = None
    height: int | None = None
    img_format = ext.lstrip(".")

    try:
        with open(p, "rb") as f:
            head = f.read(64)

            # PNG: 8-byte magic header, IHDR chunk dimensions at offset 16-24
            if head.startswith(b"\x89PNG\r\n\x1a\n") and len(head) >= 24:
                img_format = "png"
                width, height = struct.unpack(">II", head[16:24])

            # BMP: starts with 'BM', width at 18..22, height at 22..26
            elif head.startswith(b"BM") and len(head) >= 26:
                img_format = "bmp"
                width, height = struct.unpack("<II", head[18:26])

            # GIF: starts with 'GIF87a' or 'GIF89a', width at 6..8, height at 8..10
            elif head.startswith((b"GIF87a", b"GIF89a")) and len(head) >= 10:
                img_format = "gif"
                width, height = struct.unpack("<HH", head[6:10])

            # JPEG: starts with 0xFFD8
            elif head.startswith(b"\xff\xd8"):
                img_format = "jpeg"
                f.seek(2)
                buf = f.read(4096)
                idx = 0
                sof_markers = {
                    0xC0, 0xC1, 0xC2, 0xC3,
                    0xC5, 0xC6, 0xC7,
                    0xC9, 0xCA, 0xCB,
                    0xCD, 0xCE, 0xCF,
                }
                while idx < len(buf) - 9:
                    if buf[idx] == 0xFF:
                        marker = buf[idx + 1]
                        if marker in sof_markers:
                            h, w = struct.unpack(">HH", buf[idx + 5 : idx + 9])
                            height, width = h, w
                            break
                        elif marker in (0xD9, 0xDA):
                            break
                        else:
                            if idx + 3 < len(buf):
                                length = struct.unpack(">H", buf[idx + 2 : idx + 4])[0]
                                idx += 2 + length
                                continue
                    idx += 1
    except Exception:
        pass

    return {
        "path": str(p.resolve()),
        "file_name": p.name,
        "extension": ext,
        "size_bytes": size_bytes,
        "format": img_format,
        "width": width,
        "height": height,
    }


class ImageModalityPlugin(ModalityPluginPort):
    """CATML Modality Plugin for Image Data.

    Implements PluginPort and ModalityPluginPort for image modality ingestion,
    data source verification, and image metadata extraction.
    """

    def __init__(
        self,
        plugin_id: str = "image_modality",
        name: str = "Image Modality Plugin",
        version: str = "1.0.0",
        supported_extensions: tuple[str, ...] = SUPPORTED_IMAGE_EXTENSIONS,
    ) -> None:
        self.plugin_id = plugin_id
        self.name = name
        self.version = version
        self.plugin_type = PluginType.PREPROCESSOR
        self.modality = Modality.IMAGE
        self.supported_extensions = supported_extensions
        self.capabilities = PluginCapability(
            supported_tasks=["*"],
            supported_modalities=[Modality.IMAGE.value, str(Modality.IMAGE), "image", "multimodal"],
            requires_gpu=False,
            supports_proba=False,
            extra={
                "supported_extensions": list(self.supported_extensions),
                "is_multimodal": True,
                "requirements": list(self.requirements()),
                "available": self.available(),
            },
        )

    @property
    def is_available(self) -> bool:
        return self.available()

    def available(self) -> bool:
        """Checks if deep learning vision frameworks (torch, torchvision, PIL) are installed."""
        try:
            import PIL  # noqa: F401
            import torch  # noqa: F401
            import torchvision  # noqa: F401

            return True
        except ImportError:
            return False

    def requirements(self) -> tuple[str, ...]:
        """Returns optional pip dependencies required for deep learning vision features."""
        return ("catml[vision]", "torch", "torchvision", "timm", "pillow")

    def install_instructions(self) -> str:
        """User-friendly guide to install vision extras."""
        return "Install deep vision dependencies via: pip install 'catml[vision]'"

    def validate_source(self, source: DataSource) -> bool:
        """Validates that a DataSource conforms to the Image modality and references existing valid image(s).

        Args:
            source: DataSource instance to validate.

        Returns:
            True if source has Modality.IMAGE and references valid image file(s) or directories.
        """
        if not isinstance(source, DataSource):
            return False

        if source.modality != Modality.IMAGE and str(source.modality) != "image":
            return False

        if not source.path:
            return False

        path = Path(source.path)
        if not path.exists():
            return False

        if path.is_file():
            return self.validate_image_file(path)

        if path.is_dir():
            try:
                return any(
                    f.is_file() and f.suffix.lower() in self.supported_extensions
                    for f in path.iterdir()
                )
            except (OSError, PermissionError):
                return False

        return False

    def validate_image_file(self, file_path: str | Path) -> bool:
        """Validates that a single file exists and contains a supported image format.

        Args:
            file_path: Path to the image file.

        Returns:
            True if file exists, has a supported extension, and has valid header bytes.
        """
        p = Path(file_path)
        if not p.is_file():
            return False

        ext = p.suffix.lower()
        if ext not in self.supported_extensions:
            return False

        try:
            if p.stat().st_size == 0:
                return False

            with open(p, "rb") as f:
                header = f.read(16)
                if ext == ".png" and not header.startswith(b"\x89PNG\r\n\x1a\n"):
                    return False
                if ext in (".jpg", ".jpeg") and not header.startswith(b"\xff\xd8"):
                    return False
                if ext == ".bmp" and not header.startswith(b"BM"):
                    return False
                if ext == ".gif" and not header.startswith((b"GIF87a", b"GIF89a")):
                    return False
        except (OSError, PermissionError):
            return False

        return True

    def load_data(self, source: DataSource) -> dict[str, Any] | list[str]:
        """Loads and returns representations of image data for a DataSource.

        If source.path is a directory, returns a sorted list of absolute image file paths.
        If source.path is an individual file, returns a dictionary with metadata and path.

        Args:
            source: Validated DataSource.

        Returns:
            List of image paths (for directory) or metadata dictionary (for single file).

        Raises:
            ValueError: If source validation fails.
        """
        if not self.validate_source(source):
            raise ValueError(f"Invalid image DataSource '{source.id}' at path '{source.path}'.")

        p = Path(source.path)
        if p.is_dir():
            return sorted(
                str(f.resolve())
                for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in self.supported_extensions
            )

        return self.extract_metadata(p)

    def extract_metadata(self, file_path: str | Path) -> dict[str, Any]:
        """Extracts format, dimensions, and file size metadata for an image file.

        Args:
            file_path: Path to image file.

        Returns:
            Dictionary with metadata keys: path, file_name, extension, size_bytes, format, width, height.
        """
        return get_image_metadata(file_path)
