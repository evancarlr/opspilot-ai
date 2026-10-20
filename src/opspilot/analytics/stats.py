"""Small statistical helpers that SQL does not provide."""

from collections.abc import Sequence


def kaplan_meier_median(
    durations: Sequence[float], observed: Sequence[bool]
) -> float | None:
    """Median survival time, accounting for right-censored observations.

    ``observed`` is True when the event (e.g. go-live) happened at that duration,
    False when the observation was censored. Returns None if the survival curve
    never falls to 0.5, i.e. the median is not yet reached.
    """
    if len(durations) != len(observed):
        raise ValueError("durations and observed must have the same length")
    # At tied times, events are processed before censorings (standard convention).
    records = sorted(zip(durations, observed), key=lambda r: (r[0], not r[1]))
    at_risk = len(records)
    survival = 1.0
    i = 0
    while i < len(records):
        time = records[i][0]
        events = censored = 0
        while i < len(records) and records[i][0] == time:
            events += records[i][1]
            censored += not records[i][1]
            i += 1
        if events:
            survival *= 1 - events / at_risk
            if survival <= 0.5:
                return float(time)
        at_risk -= events + censored
    return None
