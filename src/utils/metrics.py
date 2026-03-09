"""
Evaluation Metrics for Cross-View Geo-Localization.

Supports: Recall@K, Average Precision (AP), Hit Rate,
          Slot Quality Metrics (NMI, cross-view consistency, entropy).
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional


def compute_recall_at_k(query_embeddings, gallery_embeddings,
                        query_labels, gallery_labels,
                        top_k: List[int] = [1, 5, 10]) -> dict:
    """
    Compute Recall@K for retrieval evaluation.

    Args:
        query_embeddings: [Nq, D] numpy array
        gallery_embeddings: [Ng, D] numpy array
        query_labels: [Nq] class labels
        gallery_labels: [Ng] class labels
        top_k: List of K values

    Returns:
        dict with recall@k values and AP
    """
    # Compute similarity matrix
    query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
    gallery_norm = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8)
    sim = np.matmul(query_norm, gallery_norm.T)  # [Nq, Ng]

    # Sort by similarity (descending)
    indices = np.argsort(-sim, axis=1)

    results = {}
    ap_sum = 0.0

    for k in top_k:
        correct = 0
        for i in range(len(query_labels)):
            top_k_indices = indices[i, :k]
            top_k_labels = gallery_labels[top_k_indices]
            if query_labels[i] in top_k_labels:
                correct += 1
        results[f'recall@{k}'] = correct / len(query_labels)

    # Average Precision (AP)
    for i in range(len(query_labels)):
        relevant = (gallery_labels[indices[i]] == query_labels[i]).astype(float)
        if relevant.sum() == 0:
            continue
        precision_at_k = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
        ap = (precision_at_k * relevant).sum() / relevant.sum()
        ap_sum += ap

    results['AP'] = ap_sum / len(query_labels)

    return results


def compute_hit_rate(query_embeddings, gallery_embeddings,
                     query_sat_indices, top_k: List[int] = [1, 5, 10]) -> dict:
    """
    Compute Hit Rate for VIGOR evaluation.

    In VIGOR, each panorama has designated positive satellite images.

    Args:
        query_embeddings: [Nq, D]
        gallery_embeddings: [Ng, D]
        query_sat_indices: list of lists, positive satellite indices per query
        top_k: K values

    Returns:
        dict with hit_rate@k values
    """
    query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
    gallery_norm = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8)
    sim = np.matmul(query_norm, gallery_norm.T)

    indices = np.argsort(-sim, axis=1)

    results = {}
    for k in top_k:
        hits = 0
        for i in range(len(query_embeddings)):
            top_k_retrieved = set(indices[i, :k].tolist())
            positives = set(query_sat_indices[i]) if isinstance(query_sat_indices[i], list) else {query_sat_indices[i]}
            if top_k_retrieved & positives:
                hits += 1
        results[f'hit_rate@{k}'] = hits / len(query_embeddings)

    return results


@torch.no_grad()
def extract_embeddings(model, dataloader, device='cuda'):
    """
    Extract embeddings from a dataset using the model.

    Args:
        model: GeoSlot model
        dataloader: DataLoader returning {'image': tensor, 'class_id': int}
        device: Device

    Returns:
        embeddings: [N, D] numpy array
        labels: [N] numpy array of class labels
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch.get('class_id', torch.zeros(images.shape[0]))

        embeddings = model.extract_embedding(images)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(np.array(labels) if isinstance(labels, (list, tuple)) else labels.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return embeddings, labels


# ============================================================================
# Slot Quality Metrics (addresses reviewer request)
# ============================================================================

def compute_slot_entropy(attn_maps: torch.Tensor, keep_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Compute entropy of slot attention maps to measure binding quality.

    Low entropy → slot focuses on a specific region (good binding)
    High entropy → slot is diffuse across the image (poor binding)

    Args:
        attn_maps: [B, K, N] slot attention maps
        keep_mask: [B, K] active slot mask

    Returns:
        dict with mean_entropy, active_slot_entropy
    """
    # Normalize to probability distribution
    attn_prob = attn_maps / (attn_maps.sum(dim=-1, keepdim=True) + 1e-8)
    log_attn = torch.log(attn_prob + 1e-8)
    entropy = -(attn_prob * log_attn).sum(dim=-1)  # [B, K]

    results = {'mean_slot_entropy': entropy.mean().item()}

    if keep_mask is not None:
        active_entropy = (entropy * keep_mask).sum() / (keep_mask.sum() + 1e-8)
        results['active_slot_entropy'] = active_entropy.item()
        results['mean_active_slots'] = keep_mask.sum(dim=-1).float().mean().item()

    return results


def compute_slot_distinctness(attn_maps: torch.Tensor, keep_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Compute pairwise overlap between slot attention maps.

    Low overlap → slots attend to distinct regions (good)
    High overlap → slots are redundant (bad)

    Args:
        attn_maps: [B, K, N]
        keep_mask: [B, K]

    Returns:
        dict with mean_overlap, max_overlap
    """
    B, K, N = attn_maps.shape
    attn_norm = attn_maps / (attn_maps.sum(dim=-1, keepdim=True) + 1e-8)

    # Pairwise cosine similarity
    pairwise = torch.bmm(attn_norm, attn_norm.transpose(1, 2))  # [B, K, K]

    # Upper triangle (excluding diagonal)
    mask_triu = torch.triu(torch.ones(K, K, device=attn_maps.device), diagonal=1).bool()
    overlaps = pairwise[:, mask_triu]  # [B, K*(K-1)/2]

    return {
        'mean_slot_overlap': overlaps.mean().item(),
        'max_slot_overlap': overlaps.max().item(),
    }


@torch.no_grad()
def compute_cross_view_slot_consistency(
    model, dataloader, device='cuda', num_batches=10
) -> Dict[str, float]:
    """
    Measure cross-view slot consistency: do matched slots (from OT) actually
    correspond to the same semantic content across views?

    Computes:
    1. Transport plan sharpness (entropy of T)
    2. Matched slot similarity (cosine sim between OT-matched pairs)
    3. Slot position consistency (do matched slots attend to similar spatial regions?)

    Args:
        model: GeoSlot model
        dataloader: DataLoader returning {'query': img, 'gallery': img}
        device: Device
        num_batches: Number of batches to evaluate

    Returns:
        dict with consistency metrics
    """
    model.eval()
    all_transport_entropy = []
    all_matched_sim = []
    all_slot_entropy_q = []
    all_slot_entropy_r = []

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        query = batch['query'].to(device)
        gallery = batch['gallery'].to(device)

        out = model(query, gallery)

        # 1. Transport plan sharpness
        T = out['transport_plan']  # [B, K, M]
        T_prob = T / (T.sum(dim=-1, keepdim=True) + 1e-8)
        T_entropy = -(T_prob * torch.log(T_prob + 1e-8)).sum(dim=-1).mean()
        all_transport_entropy.append(T_entropy.item())

        # 2. Matched slot similarity
        qs = torch.nn.functional.normalize(out['query_slots'], dim=-1)
        rs = torch.nn.functional.normalize(out['ref_slots'], dim=-1)
        # For each query slot, find best-matched ref slot via T
        best_match = T.argmax(dim=-1)  # [B, K]
        B, K, D = qs.shape
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, K)
        matched_rs = rs[batch_idx, best_match]  # [B, K, D]
        cos_sim = (qs * matched_rs).sum(dim=-1)  # [B, K]

        if out.get('query_keep_mask') is not None:
            km = out['query_keep_mask']
            sim_val = (cos_sim * km).sum() / (km.sum() + 1e-8)
        else:
            sim_val = cos_sim.mean()
        all_matched_sim.append(sim_val.item())

        # 3. Slot attention entropy
        if 'query_attn_maps' in out:
            se_q = compute_slot_entropy(out['query_attn_maps'], out.get('query_keep_mask'))
            se_r = compute_slot_entropy(out['ref_attn_maps'], out.get('ref_keep_mask'))
            all_slot_entropy_q.append(se_q['mean_slot_entropy'])
            all_slot_entropy_r.append(se_r['mean_slot_entropy'])

    results = {
        'transport_entropy': np.mean(all_transport_entropy) if all_transport_entropy else 0.0,
        'matched_slot_cosine': np.mean(all_matched_sim) if all_matched_sim else 0.0,
    }
    if all_slot_entropy_q:
        results['query_slot_entropy'] = np.mean(all_slot_entropy_q)
        results['ref_slot_entropy'] = np.mean(all_slot_entropy_r)

    return results


@torch.no_grad()
def compute_flops_and_throughput(model, input_size=(3, 224, 224), batch_size=1, device='cuda', warmup=5, repeats=20):
    """
    Measure inference throughput and memory usage.

    Args:
        model: GeoSlot model
        input_size: (C, H, W)
        batch_size: Batch size for throughput test
        device: Device
        warmup: Number of warmup iterations
        repeats: Number of timed iterations

    Returns:
        dict with throughput (samples/sec), latency (ms), peak memory (MB)
    """
    import time

    model.eval().to(device)
    dummy = torch.randn(batch_size, *input_size, device=device)

    # Warmup
    for _ in range(warmup):
        _ = model.extract_embedding(dummy)

    if device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed runs
    start = time.perf_counter()
    for _ in range(repeats):
        _ = model.extract_embedding(dummy)
    if device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_samples = batch_size * repeats
    throughput = total_samples / elapsed
    latency_ms = (elapsed / repeats) * 1000

    results = {
        'throughput_samples_per_sec': round(throughput, 1),
        'latency_ms': round(latency_ms, 2),
        'batch_size': batch_size,
    }

    if device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'):
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        results['peak_memory_mb'] = round(peak_mem, 1)

    return results
