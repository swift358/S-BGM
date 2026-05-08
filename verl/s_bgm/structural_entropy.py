from __future__ import annotations

import heapq
import math
from typing import Iterable

import numpy as np


def bipartite_to_adjacency(B: np.ndarray) -> np.ndarray:
    B = np.asarray(B, dtype=float)
    if B.ndim != 2:
        raise ValueError("B must be a bi-adjacency matrix.")
    m, n = B.shape
    A = np.zeros((m + n, m + n), dtype=float)
    A[:m, m:] = B
    A[m:, :m] = B.T
    return A


def _entropy_term(p: float) -> float:
    return 0.0 if p <= 0 else -p * math.log2(p)


def _connected_components(A: np.ndarray) -> list[list[int]]:
    visited = [False] * A.shape[0]
    components: list[list[int]] = []
    for start in range(A.shape[0]):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp: list[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.where(A[u] > 0)[0]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(int(v))
        components.append(comp)
    return components


def h1_connected(A: np.ndarray) -> float:
    degrees = A.sum(axis=1)
    volume = float(degrees.sum())
    if volume <= 0:
        return 0.0
    return float(sum(_entropy_term(float(d / volume)) for d in degrees))


def hp_connected(A: np.ndarray, partition: Iterable[Iterable[int]]) -> float:
    degrees = A.sum(axis=1)
    volume = float(degrees.sum())
    if volume <= 0:
        return 0.0

    entropy = 0.0
    all_nodes = np.arange(A.shape[0])
    for block in partition:
        S = np.array(list(block), dtype=int)
        if S.size == 0:
            continue
        block_volume = float(degrees[S].sum())
        if block_volume <= 0:
            continue
        for i in S:
            d_i = float(degrees[i])
            if d_i > 0:
                entropy -= (d_i / volume) * math.log2(d_i / block_volume)

        outside_mask = np.ones(A.shape[0], dtype=bool)
        outside_mask[S] = False
        cut = float(A[np.ix_(S, all_nodes[outside_mask])].sum())
        if cut > 0:
            entropy -= (cut / volume) * math.log2(block_volume / volume)
    return float(entropy)


def _module_score(volume: float, cut: float, total_volume: float) -> float:
    if volume <= 0:
        return 0.0
    score = volume * math.log2(volume)
    if cut > 0:
        score -= cut * math.log2(volume / total_volume)
    return score


def _merge_delta(v_a: float, g_a: float, v_b: float, g_b: float, w_ab: float, total_volume: float) -> float:
    v_new = v_a + v_b
    g_new = g_a + g_b - 2.0 * w_ab
    if g_new < 0 and abs(g_new) < 1e-12:
        g_new = 0.0
    before = _module_score(v_a, g_a, total_volume) + _module_score(v_b, g_b, total_volume)
    after = _module_score(v_new, g_new, total_volume)
    return (after - before) / total_volume


def h2_connected(A: np.ndarray) -> tuple[float, list[list[int]]]:
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    total_volume = float(A.sum())
    if total_volume <= 0:
        return 0.0, [list(range(n))]
    if n == 1:
        return 0.0, [[0]]

    max_comms = 2 * n + 5
    degrees = A.sum(axis=1)
    volumes = np.zeros(max_comms, dtype=float)
    cuts = np.zeros(max_comms, dtype=float)
    active = np.zeros(max_comms, dtype=bool)
    W = np.zeros((max_comms, max_comms), dtype=float)
    W[:n, :n] = A
    members: list[list[int] | None] = [None for _ in range(max_comms)]

    for i in range(n):
        volumes[i] = degrees[i]
        cuts[i] = degrees[i]
        active[i] = True
        members[i] = [i]

    heap: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                heapq.heappush(heap, (_merge_delta(volumes[i], cuts[i], volumes[j], cuts[j], W[i, j], total_volume), i, j))

    next_id = n
    while heap:
        _, a, b = heapq.heappop(heap)
        if not active[a] or not active[b]:
            continue
        delta = _merge_delta(volumes[a], cuts[a], volumes[b], cuts[b], W[a, b], total_volume)
        if delta >= -1e-12:
            break

        new_id = next_id
        next_id += 1
        volumes[new_id] = volumes[a] + volumes[b]
        cuts[new_id] = cuts[a] + cuts[b] - 2.0 * W[a, b]
        if cuts[new_id] < 0 and abs(cuts[new_id]) < 1e-12:
            cuts[new_id] = 0.0
        members[new_id] = list(members[a] or []) + list(members[b] or [])
        active[a] = False
        active[b] = False
        active[new_id] = True

        for c in range(next_id):
            if not active[c] or c == new_id:
                continue
            W[new_id, c] = W[a, c] + W[b, c]
            W[c, new_id] = W[new_id, c]
            if W[new_id, c] > 0:
                heapq.heappush(
                    heap,
                    (_merge_delta(volumes[new_id], cuts[new_id], volumes[c], cuts[c], W[new_id, c], total_volume), new_id, c),
                )

    partition = [sorted(members[i] or []) for i in range(next_id) if active[i]]
    partition.sort(key=lambda x: x[0] if x else -1)
    return hp_connected(A, partition), partition


def structural_entropy(A: np.ndarray, K: int = 2) -> dict[str, object]:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if np.any(A < 0):
        raise ValueError("A must contain non-negative weights.")
    if not np.allclose(A, A.T):
        raise ValueError("A must be symmetric.")
    if not np.allclose(np.diag(A), 0):
        raise ValueError("A must not contain self-loops.")
    if K not in (1, 2):
        raise ValueError("S-BGM experiments use K=1 or K=2; K=2 is the paper default.")

    total_volume = float(A.sum())
    if total_volume <= 0:
        return {"H1": 0.0, "H2": 0.0 if K == 2 else None, "best_partition": [] if K == 2 else None}

    H1 = 0.0
    H2 = 0.0
    global_partition: list[list[int]] = []
    for comp in _connected_components(A):
        A_c = A[np.ix_(comp, comp)]
        vol_c = float(A_c.sum())
        if vol_c <= 0:
            continue
        weight = vol_c / total_volume
        H1 += weight * h1_connected(A_c)
        if K == 2:
            h2, local_partition = h2_connected(A_c)
            H2 += weight * h2
            for block in local_partition:
                global_partition.append([int(comp[i]) for i in block])

    global_partition.sort(key=lambda x: x[0] if x else -1)
    return {"H1": float(H1), "H2": float(H2) if K == 2 else None, "best_partition": global_partition if K == 2 else None}


def bipartite_structural_entropy(B: np.ndarray, K: int = 2) -> dict[str, object]:
    return structural_entropy(bipartite_to_adjacency(B), K=K)

