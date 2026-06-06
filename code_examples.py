"""
code_examples.py
----------------
Catalogue of valid and violating code strings for executor testing.
Each entry is a dict with: id, description, code, expected_result_type (for valid)
or expected_failure_layer (for violating).
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
        "description": "Rolling 20-day mean of a return series",
        "artefacts": ["returns"],
        "code": """
result = returns.rolling(20).mean().dropna()
""",
        "expected_result_type": pd.Series,
    },
    {
        "id": "VC-02",
        "description": "Pairwise correlation of a position history DataFrame",
        "artefacts": ["position_history"],
        "code": """
result = position_history.iloc[:, :10].corr()
""",
        "expected_result_type": pd.DataFrame,
    },
    {
        "id": "VC-03",
        "description": "GroupBy month + mean return",
        "artefacts": ["returns"],
        "code": """
monthly = returns.copy()
monthly.index = monthly.index.to_period('M')
result = monthly.groupby(monthly.index).mean().to_frame("mean_return")
""",
        "expected_result_type": pd.DataFrame,
    },
    {
        "id": "VC-04",
        "description": "Boolean filter — count days with positive returns",
        "artefacts": ["returns"],
        "code": """
result = int((returns > 0).sum())
""",
        "expected_result_type": int,
    },
    {
        "id": "VC-05",
        "description": "Merge returns and benchmark on index",
        "artefacts": ["returns", "benchmark"],
        "code": """
combined = returns.to_frame("returns").join(
    benchmark.to_frame("benchmark"), how="inner"
)
result = combined
""",
        "expected_result_type": pd.DataFrame,
    },
    {
        "id": "VC-06",
        "description": "Cumulative return index (growth of 1)",
        "artefacts": ["returns"],
        "code": """
result = (1 + returns).cumprod()
""",
        "expected_result_type": pd.Series,
    },
    {
        "id": "VC-07",
        "description": "Descriptive statistics summary",
        "artefacts": ["returns"],
        "code": """
result = returns.describe().to_frame()
""",
        "expected_result_type": pd.DataFrame,
    },
    {
        "id": "VC-08",
        "description": "numpy corrcoef between two return series",
        "artefacts": ["returns", "benchmark"],
        "code": """
aligned = returns.align(benchmark, join="inner")
result = np.corrcoef(aligned[0].values, aligned[1].values)
""",
        "expected_result_type": np.ndarray,
    },
    {
        "id": "VC-09",
        "description": "Multi-step: compute drawdown, find worst period",
        "artefacts": ["returns"],
        "code": """
cum = (1 + returns).cumprod()
rolling_max = cum.expanding().max()
drawdown = (cum - rolling_max) / rolling_max
result = float(drawdown.min())
""",
        "expected_result_type": float,
    },
    {
        "id": "VC-10",
        "description": "Print statement captured in stdout; result still returned",
        "artefacts": ["returns"],
        "code": """
annualised_vol = returns.std() * (252 ** 0.5)
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
        "artefacts": ["returns"],
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
        "artefacts": ["returns"],
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
        "artefacts": ["returns"],
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
        "artefacts": ["returns"],
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
        "artefacts": ["returns"],
        "code": """
result = eval("1 + 1")
""",
        "expected_failure_layer": "AST",
        "expected_error_contains": "eval",
    },
    {
        "id": "VL-06",
        "description": "exec() with nested import",
        "artefacts": ["returns"],
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
        "artefacts": ["returns"],
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
        "artefacts": ["returns"],
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
        "artefacts": ["returns"],
        "code": """
x = returns.mean()
# forgot to assign result
""",
        "expected_failure_layer": "runtime",
        "expected_error_contains": "result",
    },
    {
        "id": "VL-10",
        "description": "Infinite loop — must be killed by timeout",
        "artefacts": ["returns"],
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
