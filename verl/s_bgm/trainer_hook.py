from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from ..protocol import DataProto
from .cognitive_graph import CognitiveBipartiteGraph
from .semantic_space import SemanticClusterMapper
from .stability import StabilityEstimator
from .uncertainty import extract_question, majority_vote, uncertainty_interval


class SBGMTrainerHook:
    """Driver-side hook that maps interactions to G and scales GRPO advantages."""

    def __init__(self, config: Any, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.role = getattr(config, "role", "solver")
        self.graph_path = Path(getattr(config, "graph_state_path", "sbgm/cognitive_graph/state.json"))
        self.mapper = SemanticClusterMapper(
            centroids_path=getattr(config, "centroids_path", "sbgm/semantic_space/centroids.npy"),
            embedding_model=getattr(config, "embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
            trust_remote_code=getattr(config, "trust_remote_code", True),
            allow_hash_fallback=getattr(config, "allow_hash_fallback", False),
            fallback_num_clusters=getattr(config, "semantic_clusters_m", 100),
        )
        self.graph = CognitiveBipartiteGraph.load_or_create(
            self.graph_path,
            num_question_states=getattr(config, "semantic_clusters_m", 100),
            num_solver_states=getattr(config, "uncertainty_intervals_n", 10),
            window_size=getattr(config, "sliding_window_L", 1024),
            entropy_dim=getattr(config, "structural_entropy_K", 2),
        )
        initial_entropy = getattr(config, "initial_entropy", None)
        if initial_entropy is None:
            initial_entropy = self.graph.entropy_value()
        self.stability = StabilityEstimator(
            initial_entropy=float(initial_entropy),
            entropy_decay_gamma=getattr(config, "entropy_decay_gamma", 0.01),
            pi_kp=getattr(config, "pi_kp", 1.0),
            pi_ki=getattr(config, "pi_ki", 0.01),
            alpha_min=getattr(config, "alpha_min", 0.0),
            alpha_max=getattr(config, "alpha_max", 1.0),
        )
        self.current_alpha = 0.0

    def update_from_batch(
        self,
        batch: DataProto,
        global_step: int,
        reward_metrics: dict[str, list[float]] | None = None,
    ) -> dict[str, float]:
        questions, consensus = self._extract_interactions(batch, reward_metrics or {})
        if questions:
            q_states = self.mapper.assign(questions, batch_size=getattr(self.config, "embedding_batch_size", 64))
            s_states = [
                uncertainty_interval(x, getattr(self.config, "uncertainty_intervals_n", 10))
                for x in consensus
            ]
            self.graph.add_batch(list(zip(q_states.tolist(), s_states)))
            self.graph.save(self.graph_path)

        entropy = self.graph.entropy_value()
        metrics = self.stability.update(entropy, global_step)
        self.current_alpha = metrics["alpha"]
        metrics.update(
            {
                "graph_edges": float(len(self.graph.interactions)),
                "graph_nonzero_edges": float(np.count_nonzero(self.graph.B)),
            }
        )
        return {f"s_bgm/{k}": v for k, v in metrics.items()}

    def scale_advantages(self, batch: DataProto) -> DataProto:
        if "advantages" in batch.batch:
            batch.batch["advantages"] = batch.batch["advantages"] * (1.0 - self.current_alpha)
        return batch

    def _extract_interactions(
        self,
        batch: DataProto,
        reward_metrics: dict[str, list[float]],
    ) -> tuple[list[str], list[float]]:
        responses = self._decode_responses(batch)
        if self.role == "questioner":
            questions = [extract_question(x) for x in responses]
            consensus = self._questioner_consensus_from_metrics(reward_metrics, len(questions))
            return questions, consensus
        return self._solver_interactions(batch, responses)

    def _decode_responses(self, batch: DataProto) -> list[str]:
        response_ids = batch.batch["responses"]
        response_length = batch.batch["response_mask"].sum(dim=-1)
        texts: list[str] = []
        for i in range(len(batch)):
            valid_ids = response_ids[i][: response_length[i]]
            texts.append(self.tokenizer.decode(valid_ids, skip_special_tokens=True))
        return texts

    def _decode_prompts(self, batch: DataProto) -> list[str]:
        if "source_question" in batch.non_tensor_batch:
            return [str(x) for x in batch.non_tensor_batch["source_question"]]
        if "raw_prompt_ids" not in batch.non_tensor_batch:
            return ["" for _ in range(len(batch))]
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.non_tensor_batch["raw_prompt_ids"]]

    def _questioner_consensus_from_metrics(self, reward_metrics: dict[str, list[float]], size: int) -> list[float]:
        for key in ("consensus_ratio", "score", "solver_consensus"):
            values = reward_metrics.get(key)
            if values is not None and len(values) == size:
                return [float(x) for x in values]
        scores = reward_metrics.get("overall", [])
        if len(scores) == size:
            return [0.5 + min(0.5, max(0.0, float(x))) for x in scores]
        return [0.5 for _ in range(size)]

    def _solver_interactions(self, batch: DataProto, responses: list[str]) -> tuple[list[str], list[float]]:
        prompts = self._decode_prompts(batch)
        uid = batch.non_tensor_batch.get("uid")
        if uid is None:
            grouped = {str(i): [i] for i in range(len(responses))}
        else:
            grouped: dict[str, list[int]] = defaultdict(list)
            for i, item in enumerate(uid):
                grouped[str(item)].append(i)

        questions: list[str] = []
        consensus: list[float] = []
        for indices in grouped.values():
            _, ratio, _ = majority_vote([responses[i] for i in indices])
            for i in indices:
                questions.append(prompts[i])
                consensus.append(ratio)
        return questions, consensus

