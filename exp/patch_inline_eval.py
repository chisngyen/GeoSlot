"""
patch_inline_eval.py — Inline per-altitude evaluation into ALL experiment files.
Each file becomes fully standalone — copy-paste into Kaggle and run.
No external imports needed.

Usage: python patch_inline_eval.py
"""
import os, re, glob

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_FILES = sorted(glob.glob(os.path.join(EXP_DIR, "exp[0-9]*.py")))

# ============================================================================
# The FULL inline evaluation code to replace existing evaluate + compute_metrics
# ============================================================================
INLINE_EVAL = r'''
# ============================================================================
# EVALUATION — Per-altitude R@1/R@5/R@10/mAP (paper-grade, standalone)
# ============================================================================
def compute_metrics(query_feats, gallery_feats, query_labels, gallery_labels):
    """Compute Recall@K and mAP."""
    sim_matrix = torch.mm(query_feats, gallery_feats.T)
    _, indices = sim_matrix.sort(dim=1, descending=True)
    N = query_feats.size(0)
    r1 = r5 = r10 = ap_sum = 0
    for i in range(N):
        ql = query_labels[i]
        ranked = gallery_labels[indices[i]]
        correct = torch.where(ranked == ql)[0]
        if len(correct) == 0: continue
        fc = correct[0].item()
        if fc < 1: r1 += 1
        if fc < 5: r5 += 1
        if fc < 10: r10 += 1
        ps = sum((j+1)/(p.item()+1) for j,p in enumerate(correct))
        ap_sum += ps / len(correct)
    return {'R@1': r1/N*100, 'R@5': r5/N*100, 'R@10': r10/N*100}, ap_sum/N*100


def evaluate(model, test_dataset, device, cfg=Config):
    """Full SUES-200 evaluation: 200-image gallery + per-altitude breakdown."""
    model.eval()
    tl = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
                    num_workers=cfg.NUM_WORKERS, pin_memory=True)

    # Extract drone query features + altitudes
    all_feats, all_labels, all_alts = [], [], []
    with torch.no_grad():
        for b in tl:
            f, _ = model(b['drone'].to(device))
            all_feats.append(f.cpu())
            all_labels.append(b['label'])
            all_alts.append(b['altitude'])
    all_feats = torch.cat(all_feats)
    all_labels = torch.cat(all_labels)
    all_alts = torch.cat(all_alts)

    # Build FULL satellite gallery (ALL 200 locations = confusion data)
    tr = T.Compose([T.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)), T.ToTensor(),
                     T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    root = test_dataset.root
    sat_dir = os.path.join(root, cfg.SATELLITE_DIR)
    all_loc_ids = cfg.TRAIN_LOCS + cfg.TEST_LOCS
    sf, sl, gn = [], [], []
    for loc in [f"{l:04d}" for l in all_loc_ids]:
        sp = os.path.join(sat_dir, loc, "0.png")
        if os.path.exists(sp):
            with torch.no_grad():
                f, _ = model(tr(Image.open(sp).convert('RGB')).unsqueeze(0).to(device))
            sf.append(f.cpu())
            sl.append(test_dataset.location_to_idx[loc] if loc in test_dataset.location_to_idx else -1-len(gn))
            gn.append(loc)
    sf = torch.cat(sf); sl = torch.tensor(sl)

    # Overall metrics
    overall_r, overall_ap = compute_metrics(all_feats, sf, all_labels, sl)

    # Per-altitude metrics
    altitudes = sorted(all_alts.unique().tolist())
    per_alt = {}
    for alt in altitudes:
        mask = all_alts == alt
        if mask.sum() == 0: continue
        ar, aap = compute_metrics(all_feats[mask], sf, all_labels[mask], sl)
        per_alt[int(alt)] = {'R@1': ar['R@1'], 'R@5': ar['R@5'], 'R@10': ar['R@10'],
                             'mAP': aap, 'n': int(mask.sum())}

    # Print results
    print(f"\n{'='*75}")
    print(f"  Gallery: {len(sf)} satellite images | Queries: {len(all_feats)} drone images")
    print(f"{'='*75}")
    print(f"  {'Altitude':>8s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'#Query':>6s}")
    print(f"  {'-'*50}")
    for alt in altitudes:
        a = per_alt[int(alt)]
        print(f"  {int(alt):>6d}m  {a['R@1']:6.2f}%  {a['R@5']:6.2f}%  {a['R@10']:6.2f}%  {a['mAP']:6.2f}%  {a['n']:>6d}")
    print(f"  {'-'*50}")
    print(f"  {'Overall':>8s}  {overall_r['R@1']:6.2f}%  {overall_r['R@5']:6.2f}%  {overall_r['R@10']:6.2f}%  {overall_ap:6.2f}%  {len(all_feats):>6d}")
    print(f"{'='*75}\n")

    return overall_r, overall_ap, per_alt
'''

# ============================================================================
# Patch logic
# ============================================================================
patched = 0
for fp in EXP_FILES:
    basename = os.path.basename(fp)
    with open(fp, 'r', encoding='utf-8') as f:
        code = f.read()

    # Skip if already has inline eval (marker comment)
    if 'Per-altitude R@1/R@5/R@10/mAP (paper-grade, standalone)' in code:
        print(f"  SKIP {basename} (already has inline eval)")
        continue

    # 1) Remove old eval_utils import block if exists
    code = re.sub(
        r'\n# === PATCHED: import shared eval_utils.*?HAS_EVAL_UTILS = False\n',
        '\n', code, flags=re.DOTALL)

    # 2) Remove old run_final_evaluation function if exists
    code = re.sub(
        r'\ndef run_final_evaluation\(.*?\n(?=def |if __name__|class )',
        '\n', code, flags=re.DOTALL)

    # 3) Remove old run_final_evaluation call in main
    code = re.sub(
        r'\n    # === FINAL: Per-altitude evaluation ===\n    run_final_evaluation\([^)]+\)\n',
        '\n', code)

    # 4) Find and replace old evaluate + compute_metrics functions
    # Match from "def evaluate(" or "def compute_metrics(" to the next top-level def/class
    # We need to remove BOTH compute_metrics and evaluate, then insert new versions

    # Remove old compute_metrics
    code = re.sub(
        r'\ndef compute_metrics\(.*?\n(?=def |class |# ===)',
        '\n', code, count=1, flags=re.DOTALL)

    # Remove old evaluate  
    code = re.sub(
        r'\n(?:# === EVAL.*?\n)?def evaluate\(.*?\n(?=def |class |# ===)',
        '\n', code, count=1, flags=re.DOTALL)

    # Also remove standalone section headers like "# === EVALUATION ===" 
    code = re.sub(r'\n# === EVAL[^\n]*===\n', '\n', code)

    # 5) Insert inline eval code before the training section or WarmupCosine
    # Find the WarmupCosineScheduler class or "# === TRAINING ===" marker
    insert_markers = ['class WarmupCosineScheduler', '# === TRAINING']
    insert_pos = -1
    for marker in insert_markers:
        pos = code.find(marker)
        if pos != -1:
            # Go back to start of line  
            while pos > 0 and code[pos-1] != '\n':
                pos -= 1
            insert_pos = pos
            break

    if insert_pos == -1:
        print(f"  WARN {basename}: could not find insert point, skipping")
        continue

    code = code[:insert_pos] + INLINE_EVAL + '\n' + code[insert_pos:]

    # 6) Patch main() to use new evaluate signature: returns (r, ap, per_alt)
    # Old pattern: r,ap=evaluate(model,ted,dev)  ->  r,ap,_=evaluate(model,ted,dev,Config)
    # Also: r,ap=evaluate(model,ted,dev)  with various spacing
    code = re.sub(
        r'(\s+)r,ap=evaluate\(model,ted,dev\)',
        r'\1r,ap,_=evaluate(model,ted,dev,Config)',
        code)
    # Handle spaced versions
    code = re.sub(
        r'(\s+)r, ?ap ?= ?evaluate\(model, ?ted, ?dev\)',
        r'\1r,ap,_=evaluate(model,ted,dev,Config)',
        code)
    # Handle evaluate(model, tds, dev) -> keep as is but change return
    code = re.sub(
        r'(\s+)r,ap=evaluate\(model,tds,dev\)',
        r'\1r,ap,_=evaluate(model,tds,dev,Config)',
        code)

    # 7) Handle the evaluate_with_tta in exp16 (keep it, just fix the standard eval calls)
    # exp16 has both evaluate() and evaluate_with_tta() — we only replace evaluate()

    # 8) Fix evaluate calls in evaluate_with_tta if it exists (exp16)
    # These call the old evaluate inside, we don't touch evaluate_with_tta

    with open(fp, 'w', encoding='utf-8') as f:
        f.write(code)

    # Verify syntax
    try:
        import ast
        ast.parse(code)
        print(f"  [OK] {basename}")
    except SyntaxError as e:
        print(f"  [SYNTAX ERROR] {basename}: {e}")

    patched += 1

print(f"\nDone! Patched {patched}/{len(EXP_FILES)} files with inline per-altitude evaluation.")
print("Each file is now fully standalone -- copy-paste into Kaggle cell and run!")
