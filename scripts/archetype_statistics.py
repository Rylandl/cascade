"""Validity rate and diversity of sampled archetype designs: the numbers in docs/archetypes.md.

``uv run python scripts/archetype_statistics.py [count]`` (default 40 per archetype).
"""

import sys
from collections import Counter

import jax
import numpy as np

from cascade.design.archetypes import (
    ConventionalDesign,
    FlyingWingDesign,
    sample_designs,
    validate_design,
)


def main() -> None:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    for archetype in (FlyingWingDesign, ConventionalDesign):
        reports = [
            validate_design(d) for d in sample_designs(archetype, jax.random.PRNGKey(0), count)
        ]
        valid = [r for r in reports if r.valid]
        print(f"{archetype.__name__}: {len(valid)}/{len(reports)} valid")
        reasons = Counter(reason for r in reports for reason in r.reasons)
        for reason, n in reasons.most_common():
            print(f"  rejected {n:3d}: {reason}")
        table = np.array(
            [
                [
                    r.cruise_speed_m_s,
                    np.degrees(r.angle_of_attack_rad),
                    r.throttle,
                    r.pitch_authority_rad_s2,
                    r.roll_authority_rad_s2,
                    r.short_period_hz,
                    r.short_period_damping,
                ]
                for r in valid
            ]
        )
        labels = (
            "cruise m/s",
            "alpha deg",
            "throttle",
            "pitch auth",
            "roll auth",
            "short period Hz",
            "damping",
        )
        for column, label in enumerate(labels):
            values = table[:, column]
            print(
                f"  {label:16s} min {values.min():8.2f}  median {np.median(values):8.2f}  "
                f"max {values.max():8.2f}"
            )


if __name__ == "__main__":
    main()
