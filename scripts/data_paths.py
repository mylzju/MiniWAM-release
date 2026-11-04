"""Expose dataset paths to bootstrap/simulator tools without importing torch."""

import importlib.util
from pathlib import Path

# Loading the file directly avoids miniwam.__init__ and the model dependencies.
_source = Path(__file__).resolve().parents[1] / "miniwam/data_paths.py"
_spec = importlib.util.spec_from_file_location("_miniwam_data_paths", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

data_root = _module.data_root
data_path = _module.data_path
resolve_data_path = _module.resolve_data_path
resolve_feature_roots = _module.resolve_feature_roots

if __name__ == "__main__":
    print(data_root())
