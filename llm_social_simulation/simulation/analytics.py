from __future__ import annotations


def cooperation_rate_per_round(history) -> list[float]:
    """
    Compute cooperation rate per round.

    We treat each directed action i->j as one decision.
    cooperation_rate = (# of "C") / (total actions) for that round.
    """
    rates: list[float] = []
    for tick in history:
        total = 0
        coop = 0
        for _, per_neighbor in tick.actions.items():
            for a in per_neighbor.values():
                total += 1
                if a == "C":
                    coop += 1
        rates.append(coop / total if total else 0.0)
    return rates


def final_wealth(history) -> dict[int, int]:
    """Convenience helper: wealth snapshot from the last tick."""
    if not history:
        return {}
    return dict(history[-1].wealth)
