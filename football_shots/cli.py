import argparse
from pathlib import Path

from football_shots.loader import load
from football_shots.metrics import maximos_goleadores, porcentaje_acierto, total_goles, total_tiros
from football_shots.validator import validar_tiros


def main(argv=None):
    parser = argparse.ArgumentParser(description="Valida tiros y calcula métricas")
    parser.add_argument("--input", default="Datos/shots_laliga.csv")
    args = parser.parse_args(argv)

    df = load(Path(args.input))
    validos, errores = validar_tiros(df)

    print("Tiros válidos:", total_tiros(validos))
    print("Tiros inválidos:", len(errores))
    print("Goles:", total_goles(validos))
    print("Acierto (%):", porcentaje_acierto(validos))
    print("Máximos goleadores:")
    print(maximos_goleadores(validos))


if __name__ == "__main__":
    main()