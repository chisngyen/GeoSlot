# =============================================================================
# ABLATION STUDY: GeoSlot — Run on University-1652
# 9 configs to prove each module's contribution
# Hardware: Kaggle H100 | Self-contained
# =============================================================================
import subprocess, sys
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
for pkg in ["mambavision", "timm", "transformers", "tqdm"]:
    try: __import__(pkg)
    except ImportError: install(pkg)

import os, math, glob, json, time, gc
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# --- Load shared model (with ablation flags) ---
exec(open(os.path.join(os.path.dirname(__file__), "geoslot_model.py")).read())

# === CONFIG ===
UNI1652_ROOT = "/kaggle/input/datasets/chinguyeen/university-1652/University-1652"
OUTPUT_DIR   = "/kaggle/working"
IMG_SIZE = 384;  BATCH_SIZE = 32;  ABLATION_EPOCHS = 30;  EVAL_FREQ = 10
LR_BB = 1e-5;  LR_HEAD = 1e-4;  FREEZE_BB = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 9 ABLATION CONFIGS ===
ABLATION_CONFIGS = {
    # ID: (use_slots, use_graph, use_ot, n_register, mesh_iters, description)
    "A1_baseline":    (False, False, False, 0, 0, "MambaVision-L + GAP → Cosine (no Slot/Graph/OT)"),
    "A2_slots":       (True,  False, False, 0, 0, "+ Slot Attention only"),
    "A3_register":    (True,  False, False, 4, 0, "+ Register Slots + BG Mask"),
    "A4_graph":       (True,  True,  False, 4, 0, "+ Graph Mamba (no OT)"),
    "A5_ot_soft":     (True,  True,  True,  4, 0, "+ Sinkhorn OT (no MESH)"),
    "A6_full":        (True,  True,  True,  4, 3, "Full Pipeline (MESH on) ★ Proposed"),
    "A7_no_graph":    (True,  False, True,  4, 3, "Full - Graph Mamba (to show Graph contribution)"),
    "A8_no_register": (True,  True,  True,  0, 3, "Full - Register Slots (to show Register contribution)"),
    "A9_cosine_only": (True,  True,  False, 4, 0, "Slots + Graph → Cosine Similarity (no OT)"),
}

print("=" * 70)
print("  ABLATION STUDY: GeoSlot on University-1652")
print(f"  {len(ABLATION_CONFIGS)} configs × {ABLATION_EPOCHS} epochs each")
print(f"  Device: {DEVICE}")
print("=" * 70)


# === DATASET (reuse from Phase 2) ===
class University1652Dataset(Dataset):
    def __init__(self, root, split="train", img_size=384):
        super().__init__()
        self.split = split; self.pairs = []
        self.tf = transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor(),
             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        self.tf_aug = transforms.Compose([transforms.Resize((int(img_size*1.1),int(img_size*1.1))),
             transforms.RandomCrop((img_size,img_size)),transforms.RandomHorizontalFlip(),
             transforms.ColorJitter(0.2,0.15,0.1),transforms.ToTensor(),
             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        if split == "train": self._load_train(root)
    def _find_imgs(self, d):
        return sorted(glob.glob(os.path.join(d,"**","*.jp*g"),recursive=True))
    def _load_train(self, root):
        dd=os.path.join(root,"train","drone"); sd=os.path.join(root,"train","satellite")
        if not os.path.exists(dd): return
        for cls in sorted(os.listdir(dd)):
            dc=os.path.join(dd,cls); sc=os.path.join(sd,cls)
            if not os.path.isdir(dc) or not os.path.isdir(sc): continue
            di=self._find_imgs(dc); si=self._find_imgs(sc)
            if di and si:
                for d in di: self.pairs.append((d, si[0], cls))
        print(f"  [Train] {len(self.pairs)} pairs")
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        dp,sp,cls=self.pairs[idx]
        try: d=Image.open(dp).convert("RGB"); s=Image.open(sp).convert("RGB")
        except: d=s=Image.new("RGB",(384,384),(128,128,128))
        return {"query":self.tf_aug(d),"gallery":self.tf_aug(s)}


# === EVAL ===
@torch.no_grad()
def quick_eval(model, root, img_size, device):
    model.eval()
    tf = transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor(),
         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def extract(d):
        embs,labs=[],[]
        if not os.path.exists(d): return np.array([]),np.array([])
        for cls in sorted(os.listdir(d)):
            cd=os.path.join(d,cls)
            if not os.path.isdir(cd): continue
            for ip in sorted(glob.glob(os.path.join(cd,"**","*.jp*g"),recursive=True))[:5]:
                try:
                    img=tf(Image.open(ip).convert("RGB")).unsqueeze(0).to(device)
                    embs.append(model.extract_embedding(img).cpu().numpy()[0]); labs.append(cls)
                except: pass
        return np.array(embs),np.array(labs)
    test=os.path.join(root,"test")
    for qn in ["query_drone","drone"]:
        qd=os.path.join(test,qn)
        if os.path.exists(qd): break
    for gn in ["gallery_satellite","satellite"]:
        gd=os.path.join(test,gn)
        if os.path.exists(gd): break
    qe,ql=extract(qd); ge,gl=extract(gd)
    if len(qe)==0 or len(ge)==0: return {"R@1":0,"R@5":0}
    sim=qe@ge.T; ranks=np.argsort(-sim,axis=1)
    r = {}
    for k in [1,5,10]:
        r[f"R@{k}"]=sum(1 for i in range(len(ql)) if ql[i] in gl[ranks[i,:k]])/len(ql)
    return r


# === RUN ABLATION ===
def run_one(config_id, use_slots, use_graph, use_ot, nr, mi, desc):
    print(f"\n{'='*60}")
    print(f"  [{config_id}] {desc}")
    print(f"  slots={use_slots} graph={use_graph} ot={use_ot} nr={nr} mesh={mi}")
    print(f"{'='*60}")

    model = GeoSlot(
        use_slots=use_slots, use_graph=use_graph, use_ot=use_ot,
        nr=nr, mesh_iters=mi, frozen=FREEZE_BB > 0,
    ).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = JointLoss(s2=15, s3=25).to(DEVICE)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BB},
        {"params": [p for n,p in model.named_parameters() if not n.startswith("backbone")]
         + list(criterion.parameters()), "lr": LR_HEAD},
    ], weight_decay=0.01)
    scaler = GradScaler(enabled=DEVICE.type == "cuda")
    history = []
    gs = 0

    for epoch in range(ABLATION_EPOCHS):
        if epoch == FREEZE_BB: model.backbone.unfreeze()
        model.train(); el=ea=0; nb=0; t0=time.time()
        for batch in train_loader:
            q,g=batch["query"].to(DEVICE),batch["gallery"].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=DEVICE.type=="cuda"):
                out=model(q,g,gs); ld=criterion(out,epoch); loss=ld["total_loss"]
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update()
            el+=loss.item(); ea+=ld["accuracy"].item(); nb+=1; gs+=1
        el/=max(nb,1); ea/=max(nb,1)
        if (epoch+1)%EVAL_FREQ==0 or epoch==ABLATION_EPOCHS-1:
            m=quick_eval(model, UNI1652_ROOT, IMG_SIZE, DEVICE)
            history.append({"epoch":epoch+1,"loss":round(el,4),"acc":round(ea,4),**m})
            print(f"  E{epoch+1} | Loss={el:.4f} | Acc={ea:.1%} | R@1={m['R@1']:.2%}")

    # Cleanup
    final_m = history[-1] if history else {}
    del model, criterion, optimizer, scaler
    torch.cuda.empty_cache(); gc.collect()
    return {"config_id": config_id, "description": desc,
            "use_slots": use_slots, "use_graph": use_graph, "use_ot": use_ot,
            "n_register": nr, "mesh_iters": mi,
            "final_R@1": final_m.get("R@1", 0), "final_R@5": final_m.get("R@5", 0),
            "history": history}


def main():
    global train_loader
    print("\n[1] Loading dataset...")
    train_ds = University1652Dataset(UNI1652_ROOT, "train", IMG_SIZE)
    if len(train_ds) == 0: print("[ERROR] No data!"); return
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)

    all_results = {}
    for cid, (us, ug, uo, nr, mi, desc) in ABLATION_CONFIGS.items():
        result = run_one(cid, us, ug, uo, nr, mi, desc)
        all_results[cid] = result
        # Save incrementally
        with open(os.path.join(OUTPUT_DIR, "results_ablation.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'='*70}")
    print("  ABLATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Config':<20} {'R@1':>8} {'R@5':>8} {'Description'}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*30}")
    for cid, r in all_results.items():
        print(f"  {cid:<20} {r['final_R@1']:>7.2%} {r['final_R@5']:>7.2%} {r['description'][:40]}")
    print(f"{'='*70}")

if __name__=="__main__": main()
