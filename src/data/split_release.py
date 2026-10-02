"""Which sealed splits a record may describe, decided only by the committed G-12 freeze.

Record identities refuse the sealed split by whitelist.  Once the freeze
manifest is complete and committed (the same check the SR-22 loader makes), the
single G-12 campaign must be able to describe what it measured, so the sealed
split is released here and nowhere else.  The answer is cached per process: a
campaign verifies its freeze before it starts and the freeze cannot change
under a running process.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def released_splits() -> tuple[str, ...]:
    from data.test_access import TestAccessError, _committed_freeze_manifest  # noqa: PLC0415

    try:
        _committed_freeze_manifest()
    except TestAccessError:
        return ()
    return ("test",)


__all__ = ["released_splits"]
