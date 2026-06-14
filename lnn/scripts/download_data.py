"""
Download bacterial genomes from NCBI GenBank using accession numbers.

Usage:
    python scripts/download_data.py --output data/

The accession list file (accession_list.csv) should contain columns:
    accession, species, serotype, label

NCBI Datasets CLI tool is required:
    https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ACCESSION_LIST = Path(__file__).parent / "accession_list.csv"


def download_genome(accession: str, output_dir: Path):
    """Download a single genome from NCBI."""
    cmd = [
        "datasets", "download", "genome", "accession", accession,
        "--filename", str(output_dir / f"{accession}.zip"),
        "--no-progressbar",
    ]
    subprocess.run(cmd, check=True)

    import zipfile
    with zipfile.ZipFile(output_dir / f"{accession}.zip", "r") as zf:
        zf.extractall(output_dir / accession)


def main():
    parser = argparse.ArgumentParser(description="Download genomes from NCBI")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--accessions", default=str(ACCESSION_LIST),
                        help="Path to accession list CSV")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.accessions) as f:
        reader = csv.DictReader(f)
        accessions = [row["accession"] for row in reader]

    print(f"Downloading {len(accessions)} genomes to {output_dir}...")

    for i, acc in enumerate(accessions):
        print(f"[{i+1}/{len(accessions)}] {acc}")
        try:
            download_genome(acc, output_dir)
        except subprocess.CalledProcessError as e:
            print(f"  Warning: Failed to download {acc}: {e}", file=sys.stderr)

    print("Done. Run preprocessing pipeline to build k-mer cache.")


if __name__ == "__main__":
    main()
