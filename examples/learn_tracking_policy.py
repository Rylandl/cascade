"""Train reusable sensor-aware controllers and retain their checkpoints and evaluations.

Run ``python examples/learn_tracking_policy.py dist/learning --steps 60`` or the
installed-package equivalent ``python -m cascade.learning dist/learning --steps 60``.
Use ``--resume --steps 20`` to continue, and ``--export`` for JAX inference artifacts.
"""

from cascade.learning.__main__ import main

if __name__ == "__main__":
    main()
