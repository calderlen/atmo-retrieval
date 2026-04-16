"""
Numerical guard constants grouped by purpose and dtype context.

These are not interchangeable:
- ``F32_EPS`` is a float32-relative epsilon near unity.
- ``F32_FLOOR_RECIP`` is a small positive floor for linear reciprocals ``1/x``.
- ``F32_FLOOR_RECIPSQ`` is a larger floor for squared reciprocals ``1/x^2``.
- ``F64_FLOOR`` is a very small host-side float64 floor used for underflow guards.
"""

from __future__ import annotations

import numpy as np


# Machine epsilon for float32 around 1.0. Used for relative comparisons/tolerances.
F32_EPS = float(np.finfo(np.float32).eps)

# Safe floor for linear reciprocals in float32 code.
F32_FLOOR_RECIP = 1.0e-30

# Larger floor for expressions that square the reciprocal, e.g. 1 / sigma^2.
F32_FLOOR_RECIPSQ = 1.0e-18

# float64 underflow guard.
F64_FLOOR = 1.0e-300
