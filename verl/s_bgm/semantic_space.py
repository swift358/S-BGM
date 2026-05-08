from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

try:
    from vllm import LLM
except ImportError:  # pragma: no cover
    LLM = None


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


QUESTION_KEYS = ("question", "problem", "text", "input", "prompt")


def generate_question_id(question: str, index: int) -> str:
    digest = hashlib.md5(f"{question}_{index}".encode("utf-8")).hexdigest()
    return f"q_{digest[:12]}_{index}"


def _extract_question_text(item: dict[str, Any]) -> str:
    for key in QUESTION_KEYS:
        value = item.get(key)
        if value:
            return str(value)
    return ""


def load_question_records_from_file(path: str | Path, dataset_source: str | None = None) -> list[dict[str, Any]]:
    p = Path(path)
    dataset_source = dataset_source or p.stem
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("questions", data.get("data", []))
    if not isinstance(data, list):
        raise ValueError("Reference corpus must be a list or an object with questions/data.")

    records: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if isinstance(item, str):
            text = item
            record: dict[str, Any] = {"source": dataset_source}
        elif isinstance(item, dict):
            text = _extract_question_text(item)
            record = dict(item)
            record["source"] = item.get("source", item.get("dataset_source", dataset_source))
        else:
            continue

        if text and len(text.strip()) > 10:
            record["question"] = text.strip()
            record["question_id"] = record.get("question_id") or record.get("id") or generate_question_id(text, idx)
            record.setdefault("index_in_source", idx)
            records.append(record)
    return records


def load_question_records(path: str | Path, dataset_source: str | None = None) -> list[dict[str, Any]]:
    p = Path(path)
    if p.is_dir():
        records: list[dict[str, Any]] = []
        for json_file in sorted(p.glob("*.json")):
            file_records = load_question_records_from_file(json_file, dataset_source=json_file.stem)
            print(f"Loaded {len(file_records)} questions from {json_file}")
            records.extend(file_records)
        return records
    return load_question_records_from_file(p, dataset_source=dataset_source)


class SemanticEncoder:
    def __init__(self, model_name: str, device: str | None = None, trust_remote_code: bool = True):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for S-BGM semantic clustering.")
        self.model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code, device=device)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > batch_size,
        )


def embed_with_sentence_transformers(
    questions: list[str],
    model_name: str,
    batch_size: int = 64,
    device: str | None = None,
    trust_remote_code: bool = True,
) -> np.ndarray:
    encoder = SemanticEncoder(model_name, device=device, trust_remote_code=trust_remote_code)
    return encoder.encode(questions, batch_size=batch_size)


def embed_with_vllm(
    questions: list[str],
    model_name: str,
    batch_size: int = 128,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.85,
    embedding_dim: int | None = None,
) -> np.ndarray:
    if LLM is None:
        raise ImportError("vLLM is not installed. Use --use_vllm only in an environment with vLLM.")

    try:
        model = LLM(
            model=model_name,
            runner="pooling",
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            max_model_len=max_model_len,
        )
    except TypeError:
        model = LLM(
            model=model_name,
            task="embed",
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            max_model_len=max_model_len,
        )

    tokenizer = model.get_tokenizer()
    normalized_questions: list[str] = []
    truncated = 0
    for question in tqdm(questions, desc="Token length check"):
        if len(question) > max_model_len * 2:
            tokens = tokenizer.encode(question)
            if len(tokens) > max_model_len:
                question = tokenizer.decode(tokens[: max_model_len - 1])
                truncated += 1
        normalized_questions.append(question)
    if truncated:
        print(f"Truncated {truncated} questions longer than {max_model_len} tokens.")

    first_outputs = model.embed(normalized_questions[:1])
    first_embedding = np.asarray(first_outputs[0].outputs.embedding, dtype=np.float32)
    dim = int(embedding_dim or first_embedding.shape[0])
    fd, temp_path = tempfile.mkstemp(prefix="sbgm_embeddings_", suffix=".dat")
    os.close(fd)
    embeddings = np.memmap(temp_path, dtype="float32", mode="w+", shape=(len(normalized_questions), dim))
    embeddings[0] = first_embedding

    start = 1
    for batch_start in tqdm(range(start, len(normalized_questions), batch_size), desc="Embedding"):
        batch = normalized_questions[batch_start : batch_start + batch_size]
        outputs = model.embed(batch)
        for offset, output in enumerate(outputs):
            embeddings[batch_start + offset] = np.asarray(output.outputs.embedding, dtype=np.float32)
        embeddings.flush()

    array = np.asarray(embeddings, dtype=np.float32).copy()
    try:
        os.remove(temp_path)
    except OSError:
        pass
    return l2_normalize(array)


def embed_questions(
    questions: list[str],
    model_name: str,
    batch_size: int = 64,
    device: str | None = None,
    use_vllm: bool = False,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.85,
    trust_remote_code: bool = True,
) -> np.ndarray:
    if use_vllm:
        return embed_with_vllm(
            questions,
            model_name=model_name,
            batch_size=batch_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    return l2_normalize(
        embed_with_sentence_transformers(
            questions,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            trust_remote_code=trust_remote_code,
        )
    )


def build_semantic_space(
    corpus_path: str | Path,
    output_dir: str | Path,
    embedding_model: str,
    num_clusters: int,
    batch_size: int = 64,
    dataset_source: str | None = None,
    random_state: int = 42,
    kmeans_n_init: int = 10,
    kmeans_max_iter: int = 300,
    device: str | None = None,
    use_vllm: bool = False,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.85,
    trust_remote_code: bool = True,
) -> None:
    records = load_question_records(corpus_path, dataset_source=dataset_source)
    questions = [x["question"] for x in records]
    if len(questions) < num_clusters:
        raise ValueError(f"Need at least {num_clusters} questions, got {len(questions)}.")

    print(f"Building S-BGM semantic space from {len(questions)} questions.")
    embeddings = embed_questions(
        questions,
        model_name=embedding_model,
        batch_size=batch_size,
        device=device,
        use_vllm=use_vllm,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
    )
    kmeans = KMeans(n_clusters=num_clusters, n_init=kmeans_n_init, max_iter=kmeans_max_iter, random_state=random_state)
    labels = kmeans.fit_predict(embeddings)
    centroids = l2_normalize(kmeans.cluster_centers_.astype(np.float32))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "centroids.npy", centroids)
    np.save(out / "labels.npy", labels.astype(np.int32))
    np.save(out / "embeddings.npy", embeddings.astype(np.float32))

    metadata: list[dict[str, Any]] = []
    index_to_question_id: dict[str, str] = {}
    for idx, record in enumerate(records):
        item = dict(record)
        item["index"] = idx
        item["cluster_id"] = int(labels[idx])
        item["question_id"] = item.get("question_id") or generate_question_id(item["question"], idx)
        index_to_question_id[str(idx)] = item["question_id"]
        metadata.append(item)

    (out / "question_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "index_to_question_id.json").write_text(json.dumps(index_to_question_id, indent=2), encoding="utf-8")
    stats = {
        "num_clusters": int(num_clusters),
        "num_questions": len(questions),
        "embedding_model": embedding_model,
        "embedding_dim": int(centroids.shape[1]),
        "kmeans_inertia": float(kmeans.inertia_),
        "cluster_sizes": {str(i): int((labels == i).sum()) for i in range(num_clusters)},
        "dataset_sources": sorted({str(x.get("source", "unknown")) for x in metadata}),
    }
    (out / "cluster_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved semantic space to {out.resolve()}")


class SemanticClusterMapper:
    def __init__(
        self,
        centroids_path: str | Path,
        embedding_model: str,
        device: str | None = None,
        trust_remote_code: bool = True,
        allow_hash_fallback: bool = False,
        fallback_num_clusters: int = 100,
    ):
        self.allow_hash_fallback = allow_hash_fallback
        self.encoder: SemanticEncoder | None = None
        self.centroids: np.ndarray | None = None
        p = Path(centroids_path)
        if p.exists():
            self.centroids = l2_normalize(np.load(p).astype(np.float32))
            self.encoder = SemanticEncoder(embedding_model, device=device, trust_remote_code=trust_remote_code)
            self._num_clusters = int(self.centroids.shape[0])
        elif allow_hash_fallback:
            self._num_clusters = int(fallback_num_clusters)
        else:
            raise FileNotFoundError(f"S-BGM semantic centroids not found: {centroids_path}")

    @property
    def num_clusters(self) -> int:
        return self._num_clusters

    def assign(self, questions: list[str], batch_size: int = 64) -> np.ndarray:
        if self.centroids is None or self.encoder is None:
            return np.asarray([self._hash_cluster(q) for q in questions], dtype=np.int32)
        embeddings = l2_normalize(self.encoder.encode(questions, batch_size=batch_size))
        return np.argmax(embeddings @ self.centroids.T, axis=1).astype(np.int32)

    def _hash_cluster(self, question: str) -> int:
        digest = hashlib.sha1((question or "").encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % self._num_clusters
