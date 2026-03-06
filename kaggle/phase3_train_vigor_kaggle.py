# =============================================================================
# PHASE 3: GeoSlot — Train on VIGOR (Main Benchmark #2 — Hardest)
# Target: Same-Area HR@1 ≥ 95%, Cross-Area HR@1 ≥ 30%
# SOTA: SA ~94%, CA ~25% (GeoDTR+/AuxGeo)
# Hardware: Kaggle H100 | Self-contained
# =============================================================================
import subprocess, sys
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
for pkg in ["mambavision", "timm", "transformers", "tqdm"]:
    try: __import__(pkg)
    except ImportError: install(pkg)

import os, math, glob, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# --- Load shared model ---
exec(open(os.path.join(os.path.dirname(__file__), "GeoSlot_model.py")).read())

# === CONFIG ===
VIGOR_ROOTS = {
    "chicago":      "/kaggle/input/datasets/chinguyeen/vigor-chicago",
    "newyork":      "/kaggle/input/datasets/chinguyeen/vigor-newyork",
    "sanfrancisco": "/kaggle/input/datasets/chinguyeen/vigor-sanfrancisco",
    "seattle":      "/kaggle/input/datasets/chinguyeen/vigor-seattle",
}
TRAIN_CITIES = ["chicago", "newyork", "sanfrancisco"]  # Train on 3 cities
TEST_CITY    = "seattle"                                # Cross-area test
OUTPUT_DIR   = "/kaggle/working"
RESUME_FROM  = None  # e.g. "/kaggle/working/best_model_uni1652.pth"

SAT_SIZE  = 224;  PANO_SIZE = (512, 128)
BATCH_SIZE = 32;  EPOCHS = 80;  EVAL_FREQ = 5;  SAVE_FREQ = 10
LR_BB = 5e-6;  LR_HEAD = 5e-5;  FREEZE_BB = 3
S2_EPOCH = 25;  S3_EPOCH = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("  PHASE 3: GeoSlot — VIGOR")
print(f"  Target: Same-Area HR@1 ≥ 95%, Cross-Area HR@1 ≥ 30%")
print(f"  Train: {', '.join(TRAIN_CITIES)} | Cross-Area Test: {TEST_CITY}")
print(f"  Device: {DEVICE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")
print("=" * 70)


# === DATASET ===
class VIGORDataset(Dataset):
    """
    VIGOR: Panorama ↔ Satellite matching.
    Each panorama has GPS → matched to nearest satellite tiles.
    """
    def __init__(self, cities_roots, cities, split="train",
                 sat_size=224, pano_size=(512, 128)):
        super().__init__()
        self.split = split
        self.pairs = []

        self.sat_tf = transforms.Compose([
            transforms.Resize((sat_size, sat_size)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.sat_aug = transforms.Compose([
            transforms.Resize((int(sat_size*1.1), int(sat_size*1.1))),
            transforms.RandomCrop((sat_size, sat_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.15, 0.1), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.pano_tf = transforms.Compose([
            transforms.Resize(pano_size), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.pano_aug = transforms.Compose([
            transforms.Resize(pano_size), transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.15, 0.1), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

        for city in cities:
            root = cities_roots.get(city, "")
            if not os.path.exists(root):
                print(f"  [WARN] {city} not found: {root}"); continue
            self._load_city(root, city, split)

        print(f"  [VIGOR {split}] {len(self.pairs)} pairs from {cities}")

    def _load_city(self, root, city, split):
        """Load panorama↔satellite pairs from VIGOR directory."""
        pano_dir = os.path.join(root, "panorama")
        sat_dir = os.path.join(root, "satellite")
        split_file = os.path.join(root, "splits", f"{split}_list.txt")

        if os.path.exists(split_file):
            with open(split_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2: continue
                    pp = os.path.join(pano_dir, parts[0].strip())
                    sp = os.path.join(sat_dir, parts[1].strip())
                    if not os.path.exists(pp): pp = os.path.join(root, parts[0].strip())
                    if not os.path.exists(sp): sp = os.path.join(root, parts[1].strip())
                    self.pairs.append((pp, sp, city))
        else:
            # Fallback: match by GPS proximity or filename
            panos = sorted(glob.glob(os.path.join(pano_dir, "*.jpg")))
            sats = sorted(glob.glob(os.path.join(sat_dir, "*.png")) +
                          glob.glob(os.path.join(sat_dir, "*.jpg")))
            # Simple pairing by index (VIGOR pairs by nearest GPS)
            n = min(len(panos), len(sats))
            ratio = 0.8 if split == "train" else 0.2
            start = 0 if split == "train" else int(n * 0.8)
            end = int(n * 0.8) if split == "train" else n
            for i in range(start, end):
                if i < len(panos) and i < len(sats):
                    self.pairs.append((panos[i], sats[i], city))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pp, sp, city = self.pairs[idx]
        try:
            pano = Image.open(pp).convert("RGB")
            sat = Image.open(sp).convert("RGB")
        except:
            pano = Image.new("RGB", (512, 128), (128, 128, 128))
            sat = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.split == "train":
            pano = self.pano_aug(pano); sat = self.sat_aug(sat)
        else:
            pano = self.pano_tf(pano); sat = self.sat_tf(sat)
        return {"query": pano, "gallery": sat, "city": city, "idx": idx}


# === EVALUATION ===
@torch.no_grad()
def evaluate_vigor(model, cities_roots, eval_cities, sat_size, pano_size, device):
    """Evaluate Hit Rate@K on VIGOR (same-area and/or cross-area)."""
    model.eval()
    sat_tf = transforms.Compose([transforms.Resize((sat_size,sat_size)),transforms.ToTensor(),
             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    pano_tf = transforms.Compose([transforms.Resize(pano_size),transforms.ToTensor(),
              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    # Build test dataset
    test_ds = VIGORDataset(cities_roots, eval_cities, "test", sat_size, pano_size)
    if len(test_ds) == 0:
        return {"HR@1": 0, "HR@5": 0, "HR@10": 0}
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    q_embs, r_embs = [], []
    for batch in tqdm(test_loader, desc="Eval VIGOR", leave=False):
        qe = model.extract_embedding(batch["query"].to(device)).cpu()
        re = model.extract_embedding(batch["gallery"].to(device)).cpu()
        q_embs.append(qe); r_embs.append(re)

    q_embs = torch.cat(q_embs, 0).numpy()
    r_embs = torch.cat(r_embs, 0).numpy()
    N = len(q_embs); gt = np.arange(N)
    sim = q_embs @ r_embs.T; ranks = np.argsort(-sim, axis=1)

    results = {}
    for k in [1, 5, 10]:
        results[f"HR@{k}"] = sum(1 for i in range(N) if gt[i] in ranks[i,:k]) / N
    return results


# === TRAINING ===
def main():
    print("\n[1] Loading VIGOR...")
    train_ds = VIGORDataset(VIGOR_ROOTS, TRAIN_CITIES, "train", SAT_SIZE, PANO_SIZE)
    if len(train_ds) == 0: print("[ERROR] No data!"); return
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)

    print("\n[2] Building model...")
    model = GeoSlot(frozen=FREEZE_BB > 0).to(DEVICE)
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        model.load_state_dict(torch.load(RESUME_FROM,map_location=DEVICE)["model_state_dict"],strict=False)
        print(f"  Loaded: {RESUME_FROM}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = JointLoss(s2=S2_EPOCH, s3=S3_EPOCH).to(DEVICE)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BB},
        {"params": [p for n,p in model.named_parameters() if not n.startswith("backbone")]
         + list(criterion.parameters()), "lr": LR_HEAD},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, 1e-6)
    scaler = GradScaler(enabled=DEVICE.type == "cuda")

    log = {"dataset":"vigor","history":[]}; best_hr1 = 0; gs = 0
    print(f"\n[3] Training ({EPOCHS} epochs)...\n")

    for epoch in range(EPOCHS):
        if epoch == FREEZE_BB: model.backbone.unfreeze()
        model.train(); el=ea=0; nb=0; t0=time.time()
        for batch in tqdm(train_loader, desc=f"E{epoch+1}/{EPOCHS}", leave=False):
            q,g = batch["query"].to(DEVICE), batch["gallery"].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=DEVICE.type=="cuda"):
                out=model(q,g,gs); ld=criterion(out,epoch); loss=ld["total_loss"]
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update()
            el+=loss.item(); ea+=ld["accuracy"].item(); nb+=1; gs+=1
        scheduler.step(); el/=max(nb,1); ea/=max(nb,1)
        entry={"epoch":epoch+1,"loss":round(el,4),"acc":round(ea,4),"time":round(time.time()-t0,1)}

        if (epoch+1)%EVAL_FREQ==0 or epoch==EPOCHS-1:
            # Same-area eval (all train cities)
            print(f"\n  Eval @ epoch {epoch+1}...")
            sa_m = evaluate_vigor(model, VIGOR_ROOTS, TRAIN_CITIES, SAT_SIZE, PANO_SIZE, DEVICE)
            # Cross-area eval (unseen city)
            ca_m = evaluate_vigor(model, VIGOR_ROOTS, [TEST_CITY], SAT_SIZE, PANO_SIZE, DEVICE)
            entry["same_area"] = sa_m; entry["cross_area"] = ca_m
            print(f"  Same-Area:  HR@1={sa_m.get('HR@1',0):.2%} HR@5={sa_m.get('HR@5',0):.2%}")
            print(f"  Cross-Area: HR@1={ca_m.get('HR@1',0):.2%} HR@5={ca_m.get('HR@5',0):.2%}")
            hr1 = sa_m.get("HR@1", 0)
            if hr1 > best_hr1:
                best_hr1 = hr1
                torch.save({"epoch":epoch+1,"hr1":hr1,"model_state_dict":model.state_dict()},
                           os.path.join(OUTPUT_DIR,"best_model_vigor.pth"))
                print(f"  ★ Best Same-Area HR@1: {hr1:.2%}")

        log["history"].append(entry)
        print(f"E{epoch+1} | Loss={el:.4f} | Acc={ea:.1%} | {time.time()-t0:.0f}s")
        if (epoch+1)%SAVE_FREQ==0:
            torch.save({"epoch":epoch+1,"model_state_dict":model.state_dict()},
                       os.path.join(OUTPUT_DIR,f"ckpt_vigor_ep{epoch+1}.pth"))

    log["best_hr1_sa"]=best_hr1
    with open(os.path.join(OUTPUT_DIR,"results_vigor.json"),"w") as f: json.dump(log,f,indent=2)
    print(f"\n{'='*70}\n  Done! Best Same-Area HR@1 = {best_hr1:.2%}\n{'='*70}")

if __name__=="__main__": main()
