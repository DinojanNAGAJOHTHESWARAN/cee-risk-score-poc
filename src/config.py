from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
