import numpy as np

from util.distance_utils import compute_distance_matrix


GRAPH_VARIANT_BASELINE = "baseline"
GRAPH_VARIANT_LOCAL_UPSTREAM = "local_upstream"
SUPPORTED_GRAPH_VARIANTS = (GRAPH_VARIANT_BASELINE, GRAPH_VARIANT_LOCAL_UPSTREAM)
STAGE_ADJ_STORAGE_LAYOUT = "target_by_source"
STAGE_ADJ_NORMALIZATION = "message_passing_ready"


def _validate_coords(coords):
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (N, 2). Received shape {coords.shape}.")
    if coords.shape[0] < 2:
        raise ValueError(f"At least two nodes are required to build a graph. Received N={coords.shape[0]}.")


def _pairwise_distance_std_km(dist_matrix):
    pairwise = dist_matrix[np.triu_indices(dist_matrix.shape[0], k=1)]
    positive = pairwise[pairwise > 0.0]
    if positive.size == 0:
        raise ValueError("All pairwise distances are zero; graph construction is undefined.")
    sigma_km = float(np.std(positive))
    if not np.isfinite(sigma_km) or sigma_km <= 0.0:
        raise ValueError(f"Invalid pairwise distance standard deviation: {sigma_km}.")
    return sigma_km


def _normalize_source_to_target(source_to_target):
    out_degree = np.sum(source_to_target, axis=1, keepdims=True)
    normalized = np.zeros_like(source_to_target, dtype=np.float32)
    np.divide(source_to_target, out_degree, out=normalized, where=out_degree > 0.0)
    return normalized.astype(np.float32)


def _to_message_passing_layout(source_to_target):
    return source_to_target.T.astype(np.float32)


def save_stage_adjacency(stage_path, adj, graph_variant):
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must be a square matrix. Received shape {adj.shape}.")
    if graph_variant not in SUPPORTED_GRAPH_VARIANTS:
        raise ValueError(f"Unsupported graph_variant='{graph_variant}'. Expected one of {SUPPORTED_GRAPH_VARIANTS}.")
    np.savez(
        stage_path,
        x=adj.astype(np.float32),
        graph_variant=np.array(graph_variant, dtype="U"),
        storage_layout=np.array(STAGE_ADJ_STORAGE_LAYOUT, dtype="U"),
        normalization=np.array(STAGE_ADJ_NORMALIZATION, dtype="U"),
    )


def generate_baseline_adjacency(coords, weight_threshold, self_loop=False):
    _validate_coords(coords)
    if not np.isfinite(weight_threshold) or not (0.0 < weight_threshold < 1.0):
        raise ValueError(f"weight_threshold must be in (0, 1). Received {weight_threshold}.")

    dist_matrix = compute_distance_matrix(coords)
    sigma_km = _pairwise_distance_std_km(dist_matrix)
    source_to_target = np.exp(-(dist_matrix ** 2) / (sigma_km ** 2)).astype(np.float32)
    source_to_target[source_to_target < weight_threshold] = 0.0
    if self_loop:
        np.fill_diagonal(source_to_target, 1.0)
    else:
        np.fill_diagonal(source_to_target, 0.0)

    normalized = _normalize_source_to_target(source_to_target)
    return _to_message_passing_layout(normalized)


def compute_local_upstream_probability(coords, wind_from_degrees, valid_mask):
    _validate_coords(coords)
    if wind_from_degrees.ndim != 2:
        raise ValueError(f"wind_from_degrees must have shape (T, N). Received shape {wind_from_degrees.shape}.")
    if valid_mask.shape != wind_from_degrees.shape:
        raise ValueError(
            f"valid_mask must match wind_from_degrees shape. Received {valid_mask.shape} vs {wind_from_degrees.shape}."
        )
    if wind_from_degrees.shape[1] != coords.shape[0]:
        raise ValueError(
            f"wind_from_degrees second dimension must equal number of nodes. "
            f"Received wind_from_degrees.shape={wind_from_degrees.shape}, N={coords.shape[0]}."
        )

    x = coords[:, 0]
    y = coords[:, 1]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    phi_source_to_target = np.arctan2(dy, dx).astype(np.float32)

    theta_math = np.deg2rad((270.0 - wind_from_degrees) % 360.0).astype(np.float32)
    source_to_target_probability = np.zeros((coords.shape[0], coords.shape[0]), dtype=np.float32)

    for source_idx in range(coords.shape[0]):
        source_valid = valid_mask[:, source_idx].astype(bool)
        valid_count = int(np.sum(source_valid))
        if valid_count == 0:
            raise ValueError(f"Source node {source_idx} has no valid wind direction history for local_upstream graph.")
        source_theta = theta_math[source_valid, source_idx]
        alignment = np.maximum(0.0, np.cos(source_theta[:, np.newaxis] - phi_source_to_target[source_idx][np.newaxis, :]))
        source_to_target_probability[source_idx, :] = np.mean(alignment, axis=0, dtype=np.float32)

    np.fill_diagonal(source_to_target_probability, 0.0)
    return source_to_target_probability.astype(np.float32)


def generate_local_upstream_adjacency(coords, source_to_target_probability, top_k):
    _validate_coords(coords)
    if top_k <= 0:
        raise ValueError(f"top_k must be positive. Received top_k={top_k}.")
    expected_shape = (coords.shape[0], coords.shape[0])
    if source_to_target_probability.shape != expected_shape:
        raise ValueError(
            f"source_to_target_probability must have shape {expected_shape}. Received {source_to_target_probability.shape}."
        )

    dist_matrix = compute_distance_matrix(coords)
    sigma_km = _pairwise_distance_std_km(dist_matrix)
    distance_weight = np.exp(-(dist_matrix ** 2) / (sigma_km ** 2)).astype(np.float32)
    source_to_target = source_to_target_probability.astype(np.float32) * distance_weight
    np.fill_diagonal(source_to_target, 0.0)

    sparse_source_to_target = np.zeros_like(source_to_target, dtype=np.float32)
    num_nodes = coords.shape[0]
    active_top_k = min(int(top_k), max(num_nodes - 1, 1))
    for target_idx in range(num_nodes):
        incoming = source_to_target[:, target_idx].copy()
        incoming[target_idx] = 0.0
        positive_idx = np.flatnonzero(incoming > 0.0)
        if positive_idx.size == 0:
            continue
        ordered_positive = positive_idx[np.argsort(incoming[positive_idx])[::-1]]
        keep_idx = ordered_positive[:active_top_k]
        sparse_source_to_target[keep_idx, target_idx] = incoming[keep_idx]

    normalized = _normalize_source_to_target(sparse_source_to_target)
    return _to_message_passing_layout(normalized)
