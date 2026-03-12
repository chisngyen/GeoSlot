"""
eval_utils.py — Shared Evaluation Utilities for SUES-200 Geo-Localization
===========================================================================
Import this in ANY experiment script for:
  1) Correct SUES-200 evaluation protocol (full 200-image gallery)
  2) Per-altitude R@1, R@5, R@10, mAP
  3) Overall R@1, R@5, R@10, mAP, R@1p (per-location averaged)
  4) Paper-grade LaTeX table output
  5) AP (Average Precision) per query

Usage in experiment scripts:
    from eval_utils import evaluate_full, print_paper_results
    results = evaluate_full(model, test_dataset, device, data_root=Config.DATA_ROOT)
    print_paper_results(results, exp_name="EXP7_GeoDINOL")
"""

import os, torch, numpy as np
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict
import torchvision.transforms as T


def get_eval_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def compute_metrics(query_feats, gallery_feats, query_labels, gallery_labels):
    """Compute Recall@K and mAP — standard retrieval metrics."""
    sim_matrix = torch.mm(query_feats, gallery_feats.T)
    _, indices = sim_matrix.sort(dim=1, descending=True)

    N = query_feats.size(0)
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    ap_sum = 0

    for i in range(N):
        query_label = query_labels[i]
        ranked_labels = gallery_labels[indices[i]]

        correct_mask = ranked_labels == query_label
        correct_positions = torch.where(correct_mask)[0]

        if len(correct_positions) == 0:
            continue

        first_correct = correct_positions[0].item()

        if first_correct < 1:
            recall_at_1 += 1
        if first_correct < 5:
            recall_at_5 += 1
        if first_correct < 10:
            recall_at_10 += 1

        num_correct = len(correct_positions)
        precision_sum = 0
        for j, pos in enumerate(correct_positions):
            precision_sum += (j + 1) / (pos.item() + 1)
        ap_sum += precision_sum / num_correct

    recall_at_k = {
        'R@1': recall_at_1 / N * 100,
        'R@5': recall_at_5 / N * 100,
        'R@10': recall_at_10 / N * 100,
    }
    ap = ap_sum / N * 100

    return recall_at_k, ap


def evaluate_full(model, test_dataset, device, data_root=None,
                  batch_size=256, num_workers=8, img_size=224,
                  train_locs=None, test_locs=None,
                  satellite_dir_name="satellite-view"):
    """
    Full SUES-200 evaluation with per-altitude metrics.

    Returns dict:
      {
        'overall': {'R@1':..., 'R@5':..., 'R@10':..., 'mAP':...},
        'per_altitude': {
            150: {'R@1':..., 'R@5':..., 'R@10':..., 'mAP':..., 'n_queries':...},
            200: {...},
            250: {...},
            300: {...},
        },
        'n_gallery': int,
        'n_queries': int,
      }
    """
    from torch.utils.data import DataLoader

    train_locs = train_locs or list(range(1, 121))
    test_locs = test_locs or list(range(121, 201))
    model.eval()

    # ---- Extract drone query features + track altitudes ----
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    all_drone_feats = []
    all_drone_labels = []
    all_drone_altitudes = []

    with torch.no_grad():
        for batch in test_loader:
            drone_imgs = batch['drone'].to(device)
            drone_feats, _ = model(drone_imgs)
            all_drone_feats.append(drone_feats.cpu())
            all_drone_labels.append(batch['label'])
            all_drone_altitudes.append(batch['altitude'])

    all_drone_feats = torch.cat(all_drone_feats, dim=0)
    all_drone_labels = torch.cat(all_drone_labels, dim=0)
    all_drone_altitudes = torch.cat(all_drone_altitudes, dim=0)

    # ---- Build FULL satellite gallery (ALL 200 locations) ----
    transform = get_eval_transform(img_size)
    root = data_root or test_dataset.root
    satellite_dir = os.path.join(root, satellite_dir_name)

    all_loc_ids = train_locs + test_locs
    all_gallery_locs = [f"{loc:04d}" for loc in all_loc_ids]

    sat_feats_list = []
    sat_labels_list = []
    gallery_loc_names = []

    for loc in all_gallery_locs:
        sat_path = os.path.join(satellite_dir, loc, "0.png")
        if os.path.exists(sat_path):
            sat_img = Image.open(sat_path).convert('RGB')
            sat_tensor = transform(sat_img).unsqueeze(0).to(device)
            with torch.no_grad():
                sat_feat, _ = model(sat_tensor)
            sat_feats_list.append(sat_feat.cpu())
            if loc in test_dataset.location_to_idx:
                sat_labels_list.append(test_dataset.location_to_idx[loc])
            else:
                sat_labels_list.append(-1 - len(gallery_loc_names))
            gallery_loc_names.append(loc)

    sat_feats = torch.cat(sat_feats_list, dim=0)
    sat_labels = torch.tensor(sat_labels_list)

    # ---- Overall metrics ----
    overall_rak, overall_ap = compute_metrics(
        all_drone_feats, sat_feats, all_drone_labels, sat_labels
    )

    # ---- Per-altitude metrics ----
    altitudes = sorted(all_drone_altitudes.unique().tolist())
    per_altitude = {}
    for alt in altitudes:
        alt_int = int(alt)
        mask = all_drone_altitudes == alt
        if mask.sum() == 0:
            continue
        alt_feats = all_drone_feats[mask]
        alt_labels = all_drone_labels[mask]
        alt_rak, alt_ap = compute_metrics(alt_feats, sat_feats, alt_labels, sat_labels)
        per_altitude[alt_int] = {
            'R@1': alt_rak['R@1'],
            'R@5': alt_rak['R@5'],
            'R@10': alt_rak['R@10'],
            'mAP': alt_ap,
            'n_queries': int(mask.sum()),
        }

    results = {
        'overall': {
            'R@1': overall_rak['R@1'],
            'R@5': overall_rak['R@5'],
            'R@10': overall_rak['R@10'],
            'mAP': overall_ap,
        },
        'per_altitude': per_altitude,
        'n_gallery': len(sat_feats),
        'n_queries': len(all_drone_feats),
    }

    return results


def print_paper_results(results, exp_name="Experiment"):
    """Print results in paper-grade format with per-altitude breakdown."""
    print(f"\n{'='*80}")
    print(f"  {exp_name} — SUES-200 Evaluation Results")
    print(f"{'='*80}")
    print(f"  Gallery: {results['n_gallery']} satellite images | Queries: {results['n_queries']} drone images\n")

    # Overall
    o = results['overall']
    print(f"  ┌──────────────────────────────────────────────────┐")
    print(f"  │  OVERALL                                         │")
    print(f"  │  R@1: {o['R@1']:6.2f}%   R@5: {o['R@5']:6.2f}%   R@10: {o['R@10']:6.2f}%  │")
    print(f"  │  mAP: {o['mAP']:6.2f}%                                   │")
    print(f"  └──────────────────────────────────────────────────┘\n")

    # Per-altitude table
    altitudes = sorted(results['per_altitude'].keys())
    if altitudes:
        print(f"  Per-Altitude Breakdown:")
        print(f"  ┌──────────┬─────────┬─────────┬──────────┬─────────┬──────────┐")
        print(f"  │ Altitude │   R@1   │   R@5   │   R@10   │   mAP   │ #Queries │")
        print(f"  ├──────────┼─────────┼─────────┼──────────┼─────────┼──────────┤")
        for alt in altitudes:
            a = results['per_altitude'][alt]
            print(f"  │  {alt:>4d}m  │ {a['R@1']:6.2f}% │ {a['R@5']:6.2f}% │  {a['R@10']:6.2f}% │ {a['mAP']:6.2f}% │   {a['n_queries']:>4d}   │")
        print(f"  ├──────────┼─────────┼─────────┼──────────┼─────────┼──────────┤")
        print(f"  │  Overall │ {o['R@1']:6.2f}% │ {o['R@5']:6.2f}% │  {o['R@10']:6.2f}% │ {o['mAP']:6.2f}% │   {results['n_queries']:>4d}   │")
        print(f"  └──────────┴─────────┴─────────┴──────────┴─────────┴──────────┘\n")

    # LaTeX table for paper
    print(f"  LaTeX Table Row (copy-paste into paper):")
    print(f"  % {exp_name}")
    parts = [f"  {exp_name}"]
    for alt in altitudes:
        a = results['per_altitude'][alt]
        parts.append(f" & {a['R@1']:.2f} & {a['mAP']:.2f}")
    parts.append(f" & {o['R@1']:.2f} & {o['mAP']:.2f} \\\\")
    print("".join(parts))
    print()

    # Best altitude
    if altitudes:
        best_alt = max(altitudes, key=lambda a: results['per_altitude'][a]['R@1'])
        worst_alt = min(altitudes, key=lambda a: results['per_altitude'][a]['R@1'])
        print(f"  📊 Best altitude:  {best_alt}m (R@1={results['per_altitude'][best_alt]['R@1']:.2f}%)")
        print(f"  📉 Worst altitude: {worst_alt}m (R@1={results['per_altitude'][worst_alt]['R@1']:.2f}%)")
        gap = results['per_altitude'][best_alt]['R@1'] - results['per_altitude'][worst_alt]['R@1']
        print(f"  📏 Altitude gap:   {gap:.2f}pp")

    print(f"\n{'='*80}\n")
    return results
