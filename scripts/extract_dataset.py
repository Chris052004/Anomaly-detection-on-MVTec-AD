from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def extract_dataset(archive_path: Path, output_dir: Path) -> None:
    """Estrae l'archivio del dataset MVTec AD in output_dir, a meno che non sia già stato estratto."""
    if output_dir.exists() and any(output_dir.iterdir()):  # idempotenza: non rifare 5GB di estrazione inutilmente
        print(f"{output_dir} contiene già i dati, estrazione saltata.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as tar:
        tar.extractall(path=output_dir, filter="data")  # filter="data": estrazione sicura (Python 3.12+)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estrae l'archivio del dataset MVTec AD.")
    parser.add_argument("--archive", type=Path, default=Path("mvtec_anomaly_detection.tar.xz"))
    parser.add_argument("--output", type=Path, default=Path("data/mvtec_ad"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    extract_dataset(args.archive, args.output)


if __name__ == "__main__":  # esegue main() solo se lanciato direttamente, non se importato
    main()
