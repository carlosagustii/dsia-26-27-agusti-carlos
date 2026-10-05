"""Crea una muestra reproducible del dataset completo de tiros.

Uso:
    python scripts/crear_muestra.py --input Datos/shots_dataset_cleaned.csv
    python scripts/crear_muestra.py --input Datos/shots_dataset_cleaned.csv --n 15000 --seed 7
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera Datos/shots_sample.csv")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("Datos/shots_sample.csv"))
    parser.add_argument("--n", type=int, default=20000, help="Número de filas")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    sample = frame.sample(n=min(args.n, len(frame)), random_state=args.seed)
    sample.to_csv(args.output, index=False)
    print(f"{len(sample)} filas guardadas en {args.output}")


if __name__ == "__main__":
    main()
