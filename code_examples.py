"""
code_examples.py
----------------
Catalogue of valid and violating code strings for executor testing.
Each entry is a dict with: id, description, code, expected_result_type (for valid)
or expected_failure_layer (for violating).

Artefacts available in the test registry:
  stock_returns    DataFrame  date × instrument_id  (~1305 × 1000)
  position_history DataFrame  (effective_date, instrument_id) × 3  (~60 000 × 3)
  factor_exposures DataFrame  (effective_date, instrument_id) × 5  (~60 000 × 5)
  user_signals     DataFrame  (ISIN, signal_date) × 2  (~261 000 × 2)
  universe         DataFrame  instrument_id × 7
"""

from __future__ import annotations
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# VALID CODE EXAMPLES
# All must: pass AST check, execute without error, assign `result`
# ─────────────────────────────────────────────────────────────────────────────

VALID_EXAMPLES = [
    {
        "id": "VC-01",
        "description": "Rolling 20-day mean of a return matrix (all stocks)",
        "artefacts": ["stock_returns"],
        "code": """
result = stock_returns.rolling(20).mean().dropna()
""",
        "expected_result_type": pd.DataFrame,
    },
    {
        "id": "VC-02",
        "description": "Pairwise return correlation across a subset of 10 stocks",
        "artefacts": ["stock_returns"],
        "code": """
result = stock_returns.iloc[:, :10].corr()
""",
        "expected_result_type": pd.DataFrame,
    },
    {
        "id": "VC-03",
        "description": "GroupBy month + mean return for the first stock",
        "artefacts": ["stock_returns"],
        "code": """
s = stock_returns.iloc[:, 0].copy()
s.index = s.index.to_period('M')
result = s.groupby(s.index).mean().to_frame("mean_return")
""",
        "expected_result_type": pd.DataFrame,
    },
    {
        "id": "VC-04",
        "description": "Boolean filter — count days with positive return for first stock",
        "artefacts": ["stock_returns"],
        "code": """
result = int((stock_returns.iloc[:, 0] > 0).sum())
""",
        "expected_result_type": int,
    },
    {
        "id": "VC-05",
        "description": "Join position_history cross-section with universe metadata",
        "artefacts": ["position_history", "universe"],
        "code": """
latest_date = position_history.index.get_level_values("effective_date").max()
pos = position_history.loc[latest_date]
result = pos.join(universe[["region", "industry"]])
""",
        "expected_result_type": pd.DataFrame,
    },
    {
        "id": "VC-06",
        "description": "Cumulative return index (growth of 1) for first stock",
        "artefacts": ["stock_returns"],
        "code": """
result = (1 + stock_returns.iloc[:, 0]).cumprod()
""",
        "expected_result_type": pd.Series,
    },
    {
        "id": "VC-07",
        "description": "Descriptive statistics summary for a subset of stocks",
        "artefacts": ["stock_returns"],
        "code": """
result = stock_returns.iloc[:, :5].describe()
""",
        "expected_result_type": pd.DataFrame,
    },
    {
        "id": "VC-08",
        "description": "numpy corrcoef between two stocks",
        "artefacts": ["stock_returns"],
        "code": """
result = np.corrcoef(stock_returns.iloc[:, 0].values, stock_returns.iloc[:, 1].values)
""",
        "expected_result_type": np.ndarray,
    },
    {
        "id": "VC-09",
        "description": "Multi-step: compute drawdown, find worst level for first stock",
        "artefacts": ["stock_returns"],
        "code": """
cum = (1 + stock_returns.iloc[:, 0]).cumprod()
rolling_max = cum.expanding().max()
drawdown = (cum - rolling_max) / rolling_max
result = float(drawdown.min())
""",
        "expected_result_type": float,
    },
    {
        "id": "VC-10",
        "description": "Print statement captured in stdout; result still returned",
        "artefacts": ["stock_returns"],
        "code": """
annualised_vol = stock_returns.iloc[:, 0].std() * (252 ** 0.5)
print(f"Annualised vol: {annualised_vol:.4f}")
result = float(annualised_vol)
""",
        "expected_result_type": float,
        "expects_stdout": True,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# VIOLATING CODE EXAMPLES
# All must be blocked before execution
# ─────────────────────────────────────────────────────────────────────────────

VIOLATING_EXAMPLES = [
    {
        "id": "VL-01",
        "description": "Direct os import",
        "artefacts": ["stock_returns"],
        "code": """
import os
result = os.listdir(".")
""",
        "expected_failure_layer": "AST",
        "expected_error_contains": "import",
    },
    {
        "id": "VL-02",
        "description": "from pathlib import Path",
        "artefacts": ["stock_returns"],
        "code": """
from pathlib import Path
result = Path(".").exists()
""",
        "expected_failure_layer": "AST",
        "expected_error_contains": "import",
    },
    {
        "id": "VL-03",
        "description": "sys.exit via import",
        "artefacts": ["stock_returns"],
        "code": """
import sys
sys.exit(0)
""",
        "expected_failure_layer": "AST",
        "expected_error_contains": "import",
    },
    {
        "id": "VL-04",
        "description": "open() call to read filesystem",
        "artefacts": ["stock_returns"],
        "code": """
data = open("/etc/passwd").read()
result = data
""",
        "expected_failure_layer": "AST",
        "expected_error_contains": "open",
    },
    {
        "id": "VL-05",
        "description": "eval() call",
        "artefacts": ["stock_returns"],
        "code": """
result = eval("1 + 1")
""",
        "expected_failure_layer": "AST",
        "expected_error_contains": "eval",
    },
    {
        "id": "VL-06",
        "description": "exec() with nested import",
        "artefacts": ["stock_returns"],
        "code": """
exec("import socket")
result = 1
""",
        "expected_failure_layer": "AST",
        "expected_error_contains": "exec",
    },
    {
        "id": "VL-07",
        "description": "__import__ builtin escape",
        "artefacts": ["stock_returns"],
        "code": """
os = __import__("os")
result = os.getcwd()
""",
        "expected_failure_layer": "AST",
        "expected_error_contains": "__import__",
    },
    {
        "id": "VL-08",
        "description": "Dunder subclass escape via ().__class__",
        "artefacts": ["stock_returns"],
        "code": """
subclasses = ().__class__.__bases__[0].__subclasses__()
result = subclasses
""",
        "expected_failure_layer": "AST",
        "expected_error_contains": "dunder",
    },
    {
        "id": "VL-09",
        "description": "Missing result assignment",
        "artefacts": ["stock_returns"],
        "code": """
x = stock_returns.mean()
# forgot to assign result
""",
        "expected_failure_layer": "runtime",
        "expected_error_contains": "result",
    },
    {
        "id": "VL-10",
        "description": "Infinite loop — must be killed by timeout",
        "artefacts": ["stock_returns"],
        "code": """
while True:
    pass
result = 1
""",
        "expected_failure_layer": "timeout",
        "expected_error_contains": "timed out",
        "timeout_override": 3,  # short timeout for test speed
    },
    {
        "id": "VL-11",
        "description": "Request a non-existent artefact",
        "artefacts": ["this_does_not_exist"],
        "code": """
result = this_does_not_exist.mean()
""",
        "expected_failure_layer": "pre-execution",
        "expected_error_contains": "not found",
    },
]
