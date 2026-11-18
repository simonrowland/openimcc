"""Helpers for validating frozen dataclass construction."""

from __future__ import annotations

from functools import wraps


def _reject_reinitialization(record_type: type) -> type:
    init = record_type.__init__

    @wraps(init)
    def guarded_init(self, *args, **kwargs):
        if vars(self):
            raise TypeError(
                f"{record_type.__name__} instances cannot be reinitialized"
            )
        init(self, *args, **kwargs)

    record_type.__init__ = guarded_init
    return record_type
