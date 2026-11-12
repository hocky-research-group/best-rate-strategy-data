"""Put the script directories on the import path in either layout: one flat scripts/ in the
working tree, scripts/ with analysis/, fit/ and setup/ below it in the released archive."""
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for _sub in ("", "analysis", "fit", "setup"):
    _dir = _SCRIPTS / _sub
    if _dir.is_dir() and str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))
