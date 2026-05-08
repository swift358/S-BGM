from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from .structural_entropy import bipartite_structural_entropy


class CognitiveBipartiteGraph:
    """Sliding-window graph over semantic question states and uncertainty states."""

    def __init__(self, num_question_states: int, num_solver_states: int, window_size: int, entropy_dim: int = 2):
        self.num_question_states = int(num_question_states)
        self.num_solver_states = int(num_solver_states)
        self.window_size = int(window_size)
        self.entropy_dim = int(entropy_dim)
        self.B = np.zeros((self.num_question_states, self.num_solver_states), dtype=np.float64)
        self.interactions: deque[tuple[int, int]] = deque()

    def add_interaction(self, question_state: int, solver_state: int) -> None:
        q = int(question_state)
        s = int(solver_state)
        if q < 0 or q >= self.num_question_states:
            raise ValueError(f"Question state {q} is out of range.")
        if s < 0 or s >= self.num_solver_states:
            raise ValueError(f"Solver state {s} is out of range.")
        self.B[q, s] += 1.0
        self.interactions.append((q, s))
        while len(self.interactions) > self.window_size:
            old_q, old_s = self.interactions.popleft()
            self.B[old_q, old_s] -= 1.0
            if self.B[old_q, old_s] < 0 and abs(self.B[old_q, old_s]) < 1e-12:
                self.B[old_q, old_s] = 0.0

    def add_batch(self, edges: list[tuple[int, int]]) -> None:
        for edge in edges:
            self.add_interaction(*edge)

    def entropy(self) -> dict[str, object]:
        return bipartite_structural_entropy(self.B, K=self.entropy_dim)

    def entropy_value(self) -> float:
        result = self.entropy()
        return float(result["H2"] if self.entropy_dim == 2 else result["H1"] or 0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "num_question_states": self.num_question_states,
            "num_solver_states": self.num_solver_states,
            "window_size": self.window_size,
            "entropy_dim": self.entropy_dim,
            "B": self.B.tolist(),
            "interactions": list(self.interactions),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CognitiveBipartiteGraph":
        graph = cls(
            int(payload["num_question_states"]),
            int(payload["num_solver_states"]),
            int(payload["window_size"]),
            int(payload.get("entropy_dim", 2)),
        )
        graph.B = np.asarray(payload["B"], dtype=np.float64)
        graph.interactions = deque((int(q), int(s)) for q, s in payload.get("interactions", []))
        return graph

    @classmethod
    def load_or_create(
        cls,
        path: str | Path,
        num_question_states: int,
        num_solver_states: int,
        window_size: int,
        entropy_dim: int,
    ) -> "CognitiveBipartiteGraph":
        p = Path(path)
        if p.exists():
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        return cls(num_question_states, num_solver_states, window_size, entropy_dim)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

