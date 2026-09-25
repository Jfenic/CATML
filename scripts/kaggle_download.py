"""Helper script to download and extract Kaggle competition datasets."""
import argparse
from pathlib import Path
import subprocess
import sys
import zipfile


def download_competition(competition_name: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    dest = root / "competitions" / competition_name
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n[+] Descargando dataset de '{competition_name}' en '{dest}'...")
    kaggle_bin = root / ".venv" / "bin" / "kaggle"
    if not kaggle_bin.exists():
        kaggle_bin = "kaggle"

    cmd = [str(kaggle_bin), "competitions", "download", "-c", competition_name, "-p", str(dest)]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"\n[!] Error descargando la competencia. Asegúrate de haber iniciado sesión en Kaggle.", file=sys.stderr)
        sys.exit(exc.returncode)

    # Descomprimir archivos ZIP descargados
    for zip_path in dest.glob("*.zip"):
        print(f"[+] Descomprimiendo {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dest)
        zip_path.unlink()

    print(f"\n[✓] Archivos listos en: {dest}")
    for f in dest.iterdir():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    - {f.name:25s} ({size_mb:.2f} MB)")
    print()
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descargar datasets de Kaggle")
    parser.add_argument("competition", nargs="?", default="playground-series-s4e1", help="Nombre de la competencia (ej. playground-series-s4e1)")
    args = parser.parse_args()
    download_competition(args.competition)
