"""ONNX policy runner.

The rsl_rl exporter (`scripts/rsl_rl/export_policy.py`) bakes the
RunningStandardScaler into the ONNX graph as `actor(normalizer(obs))`, so
the SDK feeds raw obs and applies no extra normalization.

Note: training-side `agents/rsl_rl_ppo_cfg.py` sets
`empirical_normalization=False` and `obs_normalization=False`, so the
baked-in normalizer is the identity at the moment — but never depend on
that and never pre-normalize obs in this file.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import onnxruntime as ort

# Suppress GPU device discovery warning on systems without a GPU (e.g. Raspberry Pi)
ort.set_default_logger_severity(3)

POLICY_ONNX = Path(__file__).resolve().parent / "policy.onnx"


class DeployedPolicy:
    def __init__(self, providers: tuple[str, ...] = ("CPUExecutionProvider",)):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(POLICY_ONNX),
            sess_options=opts,
            providers=list(providers),
        )
        self._input_name = self._session.get_inputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float32)
        if x.ndim == 1:
            x = x[None, :]
        out = self._session.run(None, {self._input_name: x})[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)
