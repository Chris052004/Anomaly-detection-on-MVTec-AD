import tarfile
from pathlib import Path

import scripts.extract_dataset as extract_script


def _build_archive(tmp_path: Path) -> Path:
    source_dir = tmp_path / "source"
    (source_dir / "bottle" / "train" / "good").mkdir(parents=True)
    (source_dir / "bottle" / "train" / "good" / "000.png").write_bytes(b"fake-png-bytes")

    archive_path = tmp_path / "dataset.tar.xz"
    with tarfile.open(archive_path, "w:xz") as tar:
        tar.add(source_dir, arcname=".")
    return archive_path


def test_extract_dataset_extracts_expected_structure(tmp_path):
    archive_path = _build_archive(tmp_path)
    output_dir = tmp_path / "data" / "mvtec_ad"

    extract_script.extract_dataset(archive_path, output_dir)

    extracted_file = output_dir / "bottle" / "train" / "good" / "000.png"
    assert extracted_file.exists()
    assert extracted_file.read_bytes() == b"fake-png-bytes"


def test_extract_dataset_skips_if_output_already_populated(tmp_path):
    archive_path = _build_archive(tmp_path)
    output_dir = tmp_path / "data" / "mvtec_ad"
    output_dir.mkdir(parents=True)
    marker = output_dir / "marker.txt"
    marker.write_text("already extracted")

    extract_script.extract_dataset(archive_path, output_dir)

    assert marker.exists()
    assert not (output_dir / "bottle").exists()


def test_parse_args_defaults():
    args = extract_script.parse_args([])
    assert args.archive == Path("mvtec_anomaly_detection.tar.xz")
    assert args.output == Path("data/mvtec_ad")


def test_parse_args_overrides():
    args = extract_script.parse_args(["--archive", "custom.tar.xz", "--output", "custom_out"])
    assert args.archive == Path("custom.tar.xz")
    assert args.output == Path("custom_out")
