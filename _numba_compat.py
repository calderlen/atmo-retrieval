"""Allow RADIS imports when Numba cannot locate a writable cache."""

import os

os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")


def _patch_numba_enable_caching() -> None:
    try:
        from numba.core.dispatcher import Dispatcher
        from numba.core.caching import NullCache
    except Exception:
        return

    orig_enable_caching = getattr(Dispatcher, "enable_caching", None)
    if orig_enable_caching is None or getattr(orig_enable_caching, "_locator_patched", False):
        return

    def _enable_caching(self):
        try:
            return orig_enable_caching(self)
        except RuntimeError as exc:
            if "no locator available" not in str(exc):
                raise
            self._cache = NullCache()
            return None

    _enable_caching._locator_patched = True
    Dispatcher.enable_caching = _enable_caching


_patch_numba_enable_caching()
