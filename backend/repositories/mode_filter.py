"""
EMPTY / DEMO / REAL read-mapping (TRD §4.5.1), as a small pure utility.

Repositories accept an explicit `data_mode: str | None` filter argument
rather than deciding it themselves (repositories perform persistence only,
per TRD §8 — the decision of "which mode is currently active" belongs to the
service layer in later phases). This module provides the one pure
translation function everyone should use so the mapping is applied
consistently, not reimplemented per call site.

Canonical mapping (TRD §4.5.1):
    EMPTY -> None   (no WHERE data_mode=... clause; there are no rows of
                     either mode to exclude when the app truly has no data)
    DEMO  -> 'demo' (WHERE data_mode = 'demo')
    REAL  -> 'real' (WHERE data_mode = 'real')

`app_state.mode` values are 'EMPTY'/'DEMO'/'REAL'; row-level `data_mode`
values are only ever 'demo'/'real' — there is no 'empty' row value, and this
function is the boundary that prevents that distinction from being blurred.
"""

_MAPPING = {"EMPTY": None, "DEMO": "demo", "REAL": "real"}


def resolve_data_mode_filter(app_mode: str) -> str | None:
    """Translate an app_state.mode value into a repository data_mode filter."""
    if app_mode not in _MAPPING:
        raise ValueError(f"Unknown app_state.mode: {app_mode!r}. Expected one of {list(_MAPPING)}.")
    return _MAPPING[app_mode]
