from __future__ import annotations

import struct
from pathlib import Path
import zlib
import numpy as np
import pandas as pd
import pytest

from automl.domain.modalities.modality import DataSource, Modality
from automl.domain.pipelines.graph import NodeType, PipelineNode
from automl.domain.plugins.plugin import PluginCapability, PluginType
from automl.domain.ports import ModalityPluginPort, PluginPort
from automl.engine.vision.image_encoder import ImageEncoderNode
from automl.plugins.modalities.image_plugin import (
    SUPPORTED_IMAGE_EXTENSIONS,
    ImageModalityPlugin,
    get_image_metadata,
)


def create_test_png(path: Path, width: int = 4, height: int = 4, color: tuple[int, int, int] = (255, 0, 0)) -> Path:
    """Helper creating a minimal valid PNG image in pure Python."""
    png = b"\x89PNG\r\n\x1a\n"
    # IHDR chunk
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data))
    png += struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + ihdr_crc
    # IDAT chunk
    raw_data = b"".join(b"\x00" + bytes(color) * width for _ in range(height))
    compressed = zlib.compress(raw_data)
    idat_crc = struct.pack(">I", zlib.crc32(b"IDAT" + compressed))
    png += struct.pack(">I", len(compressed)) + b"IDAT" + compressed + idat_crc
    # IEND chunk
    iend_crc = struct.pack(">I", zlib.crc32(b"IEND"))
    png += struct.pack(">I", 0) + b"IEND" + iend_crc

    path.write_bytes(png)
    return path


def create_test_bmp(path: Path, width: int = 2, height: int = 2) -> Path:
    """Helper creating a minimal valid 24-bit BMP image in pure Python."""
    row_bytes = (width * 3 + 3) & ~3
    pixel_data = b"\x00" * (row_bytes * height)
    file_size = 54 + len(pixel_data)

    header = (
        b"BM"
        + struct.pack("<IHHI", file_size, 0, 0, 54)
        + struct.pack("<IIIHHIIIIII", 40, width, height, 1, 24, 0, len(pixel_data), 2835, 2835, 0, 0)
    )
    path.write_bytes(header + pixel_data)
    return path


def create_test_jpeg(path: Path) -> Path:
    """Helper writing a minimal valid 1x1 JPEG image."""
    minimal_jpeg = bytes.fromhex(
        "ffd8ffe000104a46494600010101004800480000ffdb00430008060607060508070707090908"
        "0a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c303134"
        "34341f27393d38323c2e333430ffc0000b080001000101011100ffc4001f0000010501010101"
        "010100000000000000000102030405060708090a0bffda0008010100003f007f00ffd9"
    )
    path.write_bytes(minimal_jpeg)
    return path


# =====================================================================
# Tests for ImageModalityPlugin
# =====================================================================

def test_image_modality_plugin_protocol_compliance() -> None:
    plugin = ImageModalityPlugin()
    # Structural conformance with PluginPort and ModalityPluginPort
    assert hasattr(plugin, "plugin_id")
    assert hasattr(plugin, "name")
    assert hasattr(plugin, "version")
    assert hasattr(plugin, "plugin_type")
    assert hasattr(plugin, "capabilities")
    assert hasattr(plugin, "modality")
    assert callable(plugin.validate_source)
    assert callable(plugin.load_data)

    assert plugin.plugin_id == "image_modality"
    assert plugin.name == "Image Modality Plugin"
    assert plugin.version == "1.0.0"
    assert plugin.plugin_type == PluginType.PREPROCESSOR
    assert plugin.modality == Modality.IMAGE
    assert plugin.capabilities.is_compatible_with_modality("image")
    assert ".png" in plugin.capabilities.extra["supported_extensions"]


def test_image_modality_plugin_validate_source_files(tmp_path: Path) -> None:
    plugin = ImageModalityPlugin()

    png_file = create_test_png(tmp_path / "valid.png", 8, 8)
    jpeg_file = create_test_jpeg(tmp_path / "valid.jpg")
    bmp_file = create_test_bmp(tmp_path / "valid.bmp", 4, 4)

    # Valid image sources
    assert plugin.validate_source(DataSource("s1", Modality.IMAGE, str(png_file))) is True
    assert plugin.validate_source(DataSource("s2", Modality.IMAGE, str(jpeg_file))) is True
    assert plugin.validate_source(DataSource("s3", Modality.IMAGE, str(bmp_file))) is True

    # Invalid modality
    assert plugin.validate_source(DataSource("s4", Modality.TABULAR, str(png_file))) is False

    # Non-existent file
    assert plugin.validate_source(DataSource("s5", Modality.IMAGE, str(tmp_path / "absent.png"))) is False

    # Empty file
    empty_file = tmp_path / "empty.png"
    empty_file.touch()
    assert plugin.validate_source(DataSource("s6", Modality.IMAGE, str(empty_file))) is False

    # Unsupported extension
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("not an image")
    assert plugin.validate_source(DataSource("s7", Modality.IMAGE, str(txt_file))) is False

    # Fake header (corrupted file)
    corrupted_png = tmp_path / "corrupted.png"
    corrupted_png.write_bytes(b"NOT_A_PNG_FILE_HEADER")
    assert plugin.validate_source(DataSource("s8", Modality.IMAGE, str(corrupted_png))) is False


def test_image_modality_plugin_validate_source_directory(tmp_path: Path) -> None:
    plugin = ImageModalityPlugin()

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    create_test_png(img_dir / "img1.png")
    create_test_jpeg(img_dir / "img2.jpg")

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    non_img_dir = tmp_path / "text_dir"
    non_img_dir.mkdir()
    (non_img_dir / "doc.txt").write_text("text only")

    # Directory with images
    assert plugin.validate_source(DataSource("dir1", Modality.IMAGE, str(img_dir))) is True
    # Empty directory
    assert plugin.validate_source(DataSource("dir2", Modality.IMAGE, str(empty_dir))) is False
    # Directory with no images
    assert plugin.validate_source(DataSource("dir3", Modality.IMAGE, str(non_img_dir))) is False


def test_image_modality_plugin_load_data(tmp_path: Path) -> None:
    plugin = ImageModalityPlugin()

    png_file = create_test_png(tmp_path / "photo.png", width=12, height=10)
    source_file = DataSource("src_f", Modality.IMAGE, str(png_file))

    data = plugin.load_data(source_file)
    assert isinstance(data, dict)
    assert data["format"] == "png"
    assert data["width"] == 12
    assert data["height"] == 10
    assert data["size_bytes"] > 0

    # Test directory loading
    img_dir = tmp_path / "album"
    img_dir.mkdir()
    f1 = create_test_png(img_dir / "a.png")
    f2 = create_test_bmp(img_dir / "b.bmp")

    source_dir = DataSource("src_d", Modality.IMAGE, str(img_dir))
    paths = plugin.load_data(source_dir)
    assert isinstance(paths, list)
    assert len(paths) == 2
    assert str(f1.resolve()) in paths
    assert str(f2.resolve()) in paths

    # Invalid source should raise ValueError
    with pytest.raises(ValueError, match="Invalid image DataSource"):
        plugin.load_data(DataSource("bad", Modality.IMAGE, str(tmp_path / "missing")))


def test_get_image_metadata_bmp_and_jpeg(tmp_path: Path) -> None:
    bmp_path = create_test_bmp(tmp_path / "box.bmp", width=6, height=8)
    meta_bmp = get_image_metadata(bmp_path)
    assert meta_bmp["format"] == "bmp"
    assert meta_bmp["width"] == 6
    assert meta_bmp["height"] == 8

    jpg_path = create_test_jpeg(tmp_path / "point.jpg")
    meta_jpg = get_image_metadata(jpg_path)
    assert meta_jpg["format"] == "jpeg"
    assert meta_jpg["size_bytes"] > 0


# =====================================================================
# Tests for ImageEncoderNode
# =====================================================================

def test_image_encoder_initialization_and_node_conversion() -> None:
    encoder = ImageEncoderNode(node_id="vision_node", output_dim=64, model_name="deterministic")
    assert encoder.node_id == "vision_node"
    assert encoder.output_dim == 64
    assert encoder.input_modalities == (Modality.IMAGE,)
    assert encoder.output_modality == Modality.TABULAR

    # Export to domain PipelineNode
    pipe_node = encoder.to_pipeline_node()
    assert pipe_node.node_id == "vision_node"
    assert pipe_node.node_type == NodeType.ENCODER
    assert pipe_node.parameters["output_dim"] == 64
    assert pipe_node.output_modality == Modality.TABULAR

    # Reconstruct from domain PipelineNode
    restored = ImageEncoderNode.from_pipeline_node(pipe_node)
    assert restored.node_id == "vision_node"
    assert restored.output_dim == 64
    assert restored.model_name == "deterministic"


def test_image_encoder_invalid_dim() -> None:
    with pytest.raises(ValueError, match="output_dim must be strictly positive"):
        ImageEncoderNode(output_dim=0)


def test_image_encoder_deterministic_embeddings(tmp_path: Path) -> None:
    encoder = ImageEncoderNode(output_dim=32, random_seed=42)

    img1 = create_test_png(tmp_path / "red.png", color=(255, 0, 0))
    img2 = create_test_png(tmp_path / "blue.png", color=(0, 0, 255))

    # Encoding img1 twice yields exact same vector
    vec1a = encoder.encode([str(img1)])
    vec1b = encoder.encode([str(img1)])
    assert np.allclose(vec1a, vec1b)
    assert vec1a.shape == (1, 32)
    assert vec1a.dtype == np.float32

    # L2 unit normalization check
    assert np.isclose(float(np.linalg.norm(vec1a[0])), 1.0, atol=1e-5)

    # Different images produce distinct vectors
    vec2 = encoder.encode([str(img2)])
    assert not np.allclose(vec1a, vec2)

    # Cosine similarity between distinct images is strictly < 1.0
    cos_sim = float(np.dot(vec1a[0], vec2[0]))
    assert cos_sim < 0.99


def test_image_encoder_empty_input() -> None:
    encoder = ImageEncoderNode(output_dim=16)
    res = encoder.encode([])
    assert res.shape == (0, 16)
    assert res.dtype == np.float32


def test_image_encoder_batch_and_series_input(tmp_path: Path) -> None:
    encoder = ImageEncoderNode(output_dim=24, batch_size=2)

    files = [
        str(create_test_png(tmp_path / f"img_{i}.png", color=(i * 20, i * 20, 255)))
        for i in range(5)
    ]

    # Standard encode
    res_list = encoder.encode(files)
    assert res_list.shape == (5, 24)

    # Batch encode should be identical
    res_batch = encoder.encode_batch(files, batch_size=2)
    assert np.allclose(res_list, res_batch)

    # Pandas Series input
    s = pd.Series(files, name="image_paths")
    res_series = encoder.encode(s)
    assert np.allclose(res_list, res_series)


def test_image_encoder_missing_file_handling(tmp_path: Path) -> None:
    missing_path = str(tmp_path / "does_not_exist.png")

    # Policy: raise
    strict_encoder = ImageEncoderNode(output_dim=16, handle_missing="raise")
    with pytest.raises(FileNotFoundError, match="Image path does not exist"):
        strict_encoder.encode([missing_path])

    # Policy: zero
    permissive_encoder = ImageEncoderNode(output_dim=16, handle_missing="zero")
    zeros = permissive_encoder.encode([missing_path])
    assert zeros.shape == (1, 16)
    assert np.all(zeros == 0.0)


def test_image_encoder_sklearn_and_dag_interfaces(tmp_path: Path) -> None:
    encoder = ImageEncoderNode(output_dim=20)
    img_path = str(create_test_png(tmp_path / "elem.png"))

    # Sklearn transformer interface
    assert encoder.is_fitted_ is False
    encoder.fit([img_path])
    assert encoder.is_fitted_ is True

    out = encoder.transform([img_path])
    assert out.shape == (1, 20)

    out_fit_trans = encoder.fit_transform([img_path])
    assert np.allclose(out, out_fit_trans)

    # DAG execute interface with dict input
    dag_out = encoder.execute({"data": [img_path]})
    assert np.allclose(out, dag_out)


def test_image_encoder_deep_learning_fallback_warning(tmp_path: Path) -> None:
    # Requesting resnet18 should emit a UserWarning and fall back gracefully
    with pytest.warns(UserWarning, match="Vision"):
        encoder = ImageEncoderNode(output_dim=16, model_name="resnet18")
        img_path = str(create_test_png(tmp_path / "sample.png"))
        res = encoder.encode([img_path])
        assert res.shape == (1, 16)


def test_image_modality_plugin_gif_and_corrupted_headers(tmp_path: Path) -> None:
    plugin = ImageModalityPlugin()

    # GIF file
    gif_path = tmp_path / "anim.gif"
    gif_data = (
        b"GIF87a"
        + struct.pack("<HH", 7, 9)
        + b"\x80\x00\x00\xff\xff\xff\x00\x00\x00,\x00\x00\x00\x00"
        + struct.pack("<HH", 7, 9)
        + b"\x00\x02\x02D\x01\x00;"
    )
    gif_path.write_bytes(gif_data)

    assert plugin.validate_source(DataSource("g1", Modality.IMAGE, str(gif_path))) is True
    meta = plugin.extract_metadata(gif_path)
    assert meta["format"] == "gif"
    assert meta["width"] == 7
    assert meta["height"] == 9

    # Corrupted BMP
    bad_bmp = tmp_path / "bad.bmp"
    bad_bmp.write_bytes(b"NOT_A_BMP_HEADER")
    assert plugin.validate_image_file(bad_bmp) is False

    # Corrupted GIF
    bad_gif = tmp_path / "bad.gif"
    bad_gif.write_bytes(b"NOT_A_GIF_HEADER")
    assert plugin.validate_image_file(bad_gif) is False

    # Corrupted JPG
    bad_jpg = tmp_path / "bad.jpg"
    bad_jpg.write_bytes(b"NOT_A_JPG_HEADER")
    assert plugin.validate_image_file(bad_jpg) is False


def test_image_modality_plugin_invalid_inputs() -> None:
    plugin = ImageModalityPlugin()

    # Non-DataSource input
    assert plugin.validate_source("not_a_datasource") is False  # type: ignore

    # DataSource with empty path
    assert plugin.validate_source(DataSource("empty_path", Modality.IMAGE, "")) is False

    # Non-existent file metadata
    with pytest.raises(FileNotFoundError):
        plugin.extract_metadata("non_existent_file.png")


def test_image_encoder_bytes_and_numpy_inputs(tmp_path: Path) -> None:
    encoder = ImageEncoderNode(output_dim=16)

    # Direct raw bytes
    raw_bytes = b"fake_in_memory_image_bytes_content"
    vec_bytes = encoder.encode([raw_bytes])
    assert vec_bytes.shape == (1, 16)
    assert np.isclose(float(np.linalg.norm(vec_bytes[0])), 1.0, atol=1e-5)

    # Numpy array of paths
    p1 = str(create_test_png(tmp_path / "p1.png"))
    p2 = str(create_test_png(tmp_path / "p2.png"))
    np_paths = np.array([p1, p2])
    vec_np = encoder.encode(np_paths)
    assert vec_np.shape == (2, 16)

    # Single string path (auto-wrapped in list)
    vec_single = encoder.encode(p1)
    assert vec_single.shape == (1, 16)
    assert np.allclose(vec_single[0], vec_np[0])

    # DAG execute with custom dict key
    out_custom = encoder.execute({"custom_key": [p1]})
    assert np.allclose(out_custom[0], vec_single[0])


def test_image_encoder_invalid_handle_missing(tmp_path: Path) -> None:
    bad_encoder = ImageEncoderNode(output_dim=8, handle_missing="invalid_policy")
    with pytest.raises(ValueError, match="Unknown handle_missing policy"):
        bad_encoder.encode([str(tmp_path / "not_there.png")])


def test_image_encoder_base_dir_resolution(tmp_path: Path) -> None:
    sub_dir = tmp_path / "dataset" / "images"
    sub_dir.mkdir(parents=True)
    img_file = create_test_png(sub_dir / "cat.png")

    # Relative path resolution
    encoder = ImageEncoderNode(output_dim=24, base_dir=tmp_path)
    res = encoder.encode(["dataset/images/cat.png"])
    assert res.shape == (1, 24)

    # PipelineNode serialization preserves base_dir
    p_node = encoder.to_pipeline_node()
    assert p_node.parameters["base_dir"] == str(tmp_path)
    restored = ImageEncoderNode.from_pipeline_node(p_node)
    assert str(restored.base_dir) == str(tmp_path)


def test_image_encoder_caching_memory_and_disk(tmp_path: Path) -> None:
    cache_dir = tmp_path / "embeddings_cache"
    img_file = create_test_png(tmp_path / "cached_img.png")

    encoder = ImageEncoderNode(
        output_dim=32,
        use_cache=True,
        cache_dir=cache_dir,
    )

    # First encode: computes and populates memory and disk cache
    v1 = encoder.encode([str(img_file)])
    assert v1.shape == (1, 32)
    assert len(list(cache_dir.glob("*.npy"))) == 1

    # Second encode: retrieved from memory cache
    v2 = encoder.encode([str(img_file)])
    assert np.allclose(v1, v2)

    # New encoder instance pointing to same cache_dir retrieves from disk cache
    encoder2 = ImageEncoderNode(
        output_dim=32,
        use_cache=True,
        cache_dir=cache_dir,
    )
    v3 = encoder2.encode([str(img_file)])
    assert np.allclose(v1, v3)

    # Cache disabled test
    no_cache_encoder = ImageEncoderNode(output_dim=16, use_cache=False)
    no_cache_encoder.encode([str(img_file)])
    assert len(no_cache_encoder._memory_cache) == 0


def test_image_encoder_deep_learning_mocked_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    class NoGradContext:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    class MockTensor:
        def __init__(self, data: np.ndarray):
            self.data = data
        def to(self, device):
            return self
        def dim(self):
            return 2
        def cpu(self):
            return self
        def numpy(self):
            return self.data

    torch_mock = types.ModuleType("torch")
    torch_mock.device = lambda d: d
    torch_mock.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_mock.no_grad = lambda: NoGradContext()
    torch_mock.stack = lambda tensors: MockTensor(np.zeros((len(tensors), 3, 224, 224)))
    torch_mock.flatten = lambda t, dim=1: t
    torch_mock.zeros = lambda *shape: MockTensor(np.zeros(shape))
    torch_mock.nn = types.SimpleNamespace(Identity=lambda: None)

    class MockModel:
        fc = None
        def to(self, device):
            return self
        def eval(self):
            return self
        def __call__(self, batch):
            n = batch.data.shape[0] if hasattr(batch, "data") else 1
            return MockTensor(np.ones((n, 512), dtype=np.float32))

    torchvision_mock = types.ModuleType("torchvision")
    tv_models_mock = types.ModuleType("torchvision.models")
    tv_models_mock.resnet18 = lambda *a, **kw: MockModel()
    tv_transforms_mock = types.ModuleType("torchvision.transforms")
    tv_transforms_mock.Compose = lambda funcs: lambda x: MockTensor(np.zeros((3, 224, 224)))
    tv_transforms_mock.Resize = lambda s: None
    tv_transforms_mock.CenterCrop = lambda s: None
    tv_transforms_mock.ToTensor = lambda: None
    tv_transforms_mock.Normalize = lambda **kw: None

    pil_mock = types.ModuleType("PIL")
    pil_image_mock = types.ModuleType("PIL.Image")
    class DummyPILImg:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def convert(self, mode):
            return self
    pil_image_mock.open = lambda p: DummyPILImg()
    pil_mock.Image = pil_image_mock

    monkeypatch.setitem(sys.modules, "torch", torch_mock)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision_mock)
    monkeypatch.setitem(sys.modules, "torchvision.models", tv_models_mock)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", tv_transforms_mock)
    monkeypatch.setitem(sys.modules, "PIL", pil_mock)
    monkeypatch.setitem(sys.modules, "PIL.Image", pil_image_mock)

    img_file = create_test_png(tmp_path / "deep_test.png")
    encoder = ImageEncoderNode(output_dim=64, model_name="resnet18")
    embeddings = encoder.encode([str(img_file)])
    assert embeddings.shape == (1, 64)
    assert np.isclose(float(np.linalg.norm(embeddings[0])), 1.0, atol=1e-5)


