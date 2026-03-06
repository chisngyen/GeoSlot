"""
Evaluation Metrics for Cross-View Geo-Localization.

Supports: Recall@K, Average Precision (AP), Hit Rate.
"""

import numpy as np
import torch
from typing import List, Tuple


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
