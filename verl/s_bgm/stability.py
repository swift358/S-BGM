from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class StabilityEstimator:
    """Entropy target decay and PI control from S-BGM Eq. 3-4."""

    initial_entropy: float
    entropy_decay_gamma: float = 0.01
    pi_kp: float = 1.0
    pi_ki: float = 0.01
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    errors: list[float] = field(default_factory=list)

    def target_entropy(self, step: int) -> float:
        return float(self.initial_entropy - self.entropy_decay_gamma * math.log(1.0 + max(0, step)))

    def update(self, current_entropy: float, step: int) -> dict[str, float]:
        target = self.target_entropy(step)
        error = target - float(current_entropy)
        integral = sum(self.errors)
        raw_alpha = self.pi_kp * error + self.pi_ki * integral
        alpha = min(self.alpha_max, max(self.alpha_min, raw_alpha))
        self.errors.append(error)
        return {
            "target_entropy": target,
            "current_entropy": float(current_entropy),
            "entropy_error": error,
            "integral_error": integral,
            "alpha": float(alpha),
        }

