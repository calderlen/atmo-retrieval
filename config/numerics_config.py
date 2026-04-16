"""
Numerical guard constants grouped by purpose and dtype context.

These are not interchangeable:
- ``F32_EPS`` is a float32-relative epsilon near unity.
- ``F32_FLOOR_RECIP`` is a small positive floor for linear reciprocals ``1/x``.
- ``F32_FLOOR_RECIPSQ`` is a larger floor for squared reciprocals ``1/x^2``.
- ``F32_GRAVITY_FLOOR`` is a P-T denominator floor for gravity-like terms.
- ``F32_LENGTHSCALE_FLOOR`` is a GP lengthscale floor.
- ``F32_STDDEV_FLOOR`` is a small positive stabilizer for standard deviations.
- ``F64_FLOOR`` is a very small host-side float64 floor used for underflow guards.
- ``TRACE_SPECIES_FLOOR`` is a chemistry abundance/profile floor for absent trace species.
"""

from __future__ import annotations

import numpy as np


# Machine epsilon for float32 around 1.0. Used for relative comparisons/tolerances.
F32_EPS = float(np.finfo(np.float32).eps)
F32_FLOOR_RECIP = 1.0e-30 # Safe floor for linear reciprocals in float32 code.
F32_FLOOR_RECIPSQ = 1.0e-18 # Larger floor for expressions that square the reciprocal, e.g. 1 / sigma^2.
F32_GRAVITY_FLOOR = 1.0e-20 # Safe floor for gravity-like denominators in P-T profile code.
F32_LENGTHSCALE_FLOOR = 1.0e-12 # Safe floor for GP lengthscales in standardized coordinate units.
F32_STDDEV_FLOOR = 1.0e-12 # Small stabilizer for standard deviation-like scale terms.
F64_FLOOR = 1.0e-300 # float64 underflow guard.
TRACE_SPECIES_FLOOR = 1.0e-30 # Semantic floor for absent/trace chemistry species profiles.