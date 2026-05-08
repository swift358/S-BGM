#!/usr/bin/env python
from __future__ import annotations

import argparse

from verl.s_bgm.semantic_space import build_semantic_space


def main() -> None:
    parser = argparse.ArgumentParser(description="Build S-BGM semantic question states with K-means.")
    parser.add_argument("--corpus", required=True, help="JSON file or directory with question/problem/text/input/prompt fields.")
    parser.add_argument("--dataset_source", default=None, help="Source name for a single corpus file.")
    parser.add_argument("--output_dir", default="sbgm/semantic_space")
    parser.add_argument("--embedding_model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--num_clusters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--kmeans_n_init", type=int, default=10)
    parser.add_argument("--kmeans_max_iter", type=int, default=300)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--no_trust_remote_code", action="store_true")
    args = parser.parse_args()

    build_semantic_space(
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        num_clusters=args.num_clusters,
        batch_size=args.batch_size,
        dataset_source=args.dataset_source,
        random_state=args.random_state,
        kmeans_n_init=args.kmeans_n_init,
        kmeans_max_iter=args.kmeans_max_iter,
        device=args.device,
        use_vllm=args.use_vllm,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=not args.no_trust_remote_code,
    )


if __name__ == "__main__":
    main()
