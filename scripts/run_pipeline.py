"""Exécute les notebooks du projet dans l'ordre (01 -> 02..06 -> 07), sur place, avec nbclient.

Exemples :
    python scripts/run_pipeline.py                 # tout
    python scripts/run_pipeline.py --only 02 03    # seulement ces notebooks
    python scripts/run_pipeline.py --skip-prep     # tout sauf 01
Code de sortie non nul au premier échec (le notebook fautif n'est pas écrasé).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
LOG = logging.getLogger("pipeline")


def find_notebooks() -> dict[str, Path]:
    """{'01': chemin, ...} pour les notebooks `NN_*.ipynb` (hors copies *.executed.ipynb)."""
    found = {p.name[:2]: p for p in sorted(NB_DIR.glob("[0-9][0-9]_*.ipynb")) if ".executed" not in p.name}
    return found


def run_notebook(path: Path) -> float:
    """Exécute un notebook sur place ; renvoie la durée en secondes."""
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}})
    start = time.perf_counter()
    client.execute()
    nbformat.write(nb, path)
    return time.perf_counter() - start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="NN", help="numéros de notebooks à exécuter (ex. 02 03)")
    parser.add_argument("--skip-prep", action="store_true", help="ne pas rejouer 01_prep")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if sys.platform == "win32":  # pyzmq exige la boucle « selector » sous Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    available = find_notebooks()
    wanted = [k.zfill(2) for k in args.only] if args.only else list(available)
    unknown = [k for k in wanted if k not in available]
    if unknown:
        LOG.error("notebook(s) introuvable(s) : %s (disponibles : %s)", unknown, list(available))
        return 2
    if args.skip_prep:
        wanted = [k for k in wanted if k != "01"]

    total = time.perf_counter()
    for key in sorted(wanted):
        path = available[key]
        LOG.info("[%s] %s ...", key, path.name)
        try:
            duration = run_notebook(path)
        except CellExecutionError as exc:
            LOG.error("ÉCHEC du notebook %s (non modifié) :\n%s", path.name, str(exc)[-1500:])
            return 1
        LOG.info("[%s] terminé en %.1f s", key, duration)
    LOG.info("pipeline terminé : %d notebook(s) en %.1f s", len(wanted), time.perf_counter() - total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
