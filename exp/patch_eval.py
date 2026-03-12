"""
patch_eval.py — Run this ONCE to patch all 12 experiment files
with proper per-altitude evaluation from eval_utils.py.

Usage: python patch_eval.py
"""
import os, re, glob

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_FILES = sorted(glob.glob(os.path.join(EXP_DIR, "exp[0-9]*.py")))

# The evaluation upgrade snippet to inject at top (import)
IMPORT_PATCH = '''
# === PATCHED: import shared eval_utils for per-altitude evaluation ===
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')
try:
    from eval_utils import evaluate_full, print_paper_results
    HAS_EVAL_UTILS = True
except ImportError:
    HAS_EVAL_UTILS = False
'''

# The final evaluation block to append before `if __name__`
FINAL_EVAL_BLOCK = '''
def run_final_evaluation(model, test_dataset, device, exp_name, cfg=Config):
    """Run comprehensive per-altitude evaluation with paper-grade output."""
    if HAS_EVAL_UTILS:
        results = evaluate_full(
            model, test_dataset, device,
            data_root=cfg.DATA_ROOT,
            batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.NUM_WORKERS,
            img_size=cfg.IMG_SIZE,
            train_locs=cfg.TRAIN_LOCS,
            test_locs=cfg.TEST_LOCS,
        )
        print_paper_results(results, exp_name=exp_name)
        return results
    else:
        print("eval_utils not found, using basic evaluate()")
        r, ap = evaluate(model, test_dataset, device)
        print(f"R@1:{r['R@1']:.2f}% R@5:{r['R@5']:.2f}% R@10:{r['R@10']:.2f}% mAP:{ap:.2f}%")
        return {'overall': {**r, 'mAP': ap}}
'''

patched = 0
for fp in EXP_FILES:
    basename = os.path.basename(fp)
    with open(fp, 'r', encoding='utf-8') as f:
        code = f.read()

    if 'HAS_EVAL_UTILS' in code:
        print(f"  SKIP {basename} (already patched)")
        continue

    # 1) Add import after existing imports
    # Find "import os" or first import line
    import_pos = code.find("import os")
    if import_pos == -1:
        import_pos = code.find("import torch")
    if import_pos == -1:
        print(f"  SKIP {basename} (no import found)")
        continue

    # Insert import patch right after the line containing "import os"
    line_end = code.find('\n', import_pos)
    code = code[:line_end+1] + IMPORT_PATCH + code[line_end+1:]

    # 2) Add run_final_evaluation function before `if __name__`
    main_pos = code.find('if __name__')
    if main_pos == -1:
        print(f"  SKIP {basename} (no __main__)")
        continue
    code = code[:main_pos] + FINAL_EVAL_BLOCK + '\n' + code[main_pos:]

    # 3) Patch main() to call run_final_evaluation at the end
    # Find the last print statement in main that says "Done!" or "Best R@1"
    # and add the final eval call after it
    done_pattern = re.search(r'(print\(f"\\n\{.*?Done!.*?\))', code)
    if done_pattern:
        insert_pos = done_pattern.end()
        # Extract experiment name from filename
        exp_name = basename.replace('.py', '').upper().replace('_', ' ')
        final_call = f'\n    # === FINAL: Per-altitude evaluation ===\n    run_final_evaluation(model, ted, dev, "{exp_name}", Config)\n'
        code = code[:insert_pos] + final_call + code[insert_pos:]

    with open(fp, 'w', encoding='utf-8') as f:
        f.write(code)
    patched += 1
    print(f"  [OK] PATCHED {basename}")

print(f"\nDone! Patched {patched}/{len(EXP_FILES)} files.")
print("All experiments now output per-altitude R@1/R@5/R@10/mAP + LaTeX table row.")
