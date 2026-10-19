import importlib.util
from pathlib import Path
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "preliminary_threshold_calibration.py"
spec = importlib.util.spec_from_file_location("threshold_calibration", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_historical_threshold_matrix_shapes_and_minima():
    assert module.CONSTANT_RMSE.shape == (4, 5)
    assert module.VARIABLE_RMSE.shape == (4, 5)
    # Historical heatmaps identify 0.93 / 0.60 as the development compromise.
    assert np.isclose(module.CONSTANT_RMSE[0, 2], 1.724)
    assert np.isclose(module.VARIABLE_RMSE[0, 2], 1.821)
