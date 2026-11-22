"""Ajoute src/ au chemin d'import pour que `import ml_utils` fonctionne dans les tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
