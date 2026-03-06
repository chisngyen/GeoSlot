# =============================================================================
# PHASE 4: GeoSlot — Train on CV-Cities (Generalization Test)
# Target: Establish cross-city baseline (no prior SOTA)
# Train: 12 cities → Test: 4 unseen cities
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
exec(open(os.path.join(os.path.dirname(__file__), "geoslot_model.py")).read())

# === CONFIG ===
CVCITIES_ROOT = "/kaggle/input/datasets/chisboiz/cv-cities"
OUTPUT_DIR    = "/kaggle/working"
RESUME_FROM   = None  # e.g. "/kaggle/working/best_model_vigor.pth"

TRAIN_CITIES = ["barcelona","buenosaires","lisbon","london","melbourne",
                "mexicocity","moscow","newyork","sanfrancisco","santiago","saopaulo","toronto"]
TEST_CITIES  = ["berlin","osaka","capetown","tokyo"]  # Diverse continents

SAT_SIZE = 224;  PANO_SIZE = (512, 128)
BATCH_SIZE = 32;  EPOCHS = 40;  EVAL_FREQ = 5;  SAVE_FREQ = 10
LR_BB = 5e-6;  LR_HEAD = 5e-5;  FREEZE_BB = 3
S2_EPOCH = 15;  S3_EPOCH = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("  PHASE 4: GeoSlot — CV-Cities (Generalization)")
print(f"  Train: {len(TRAIN_CITIES)} cities | Test: {len(TEST_CITIES)} unseen cities")
print(f"  Device: {DEVICE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")
print("=" * 70)


# === DATASET ===
class CVCitiesDataset(Dataset):
    """CV-Cities: pano ↔ satellite matching across cities."""
    def __init__(self, root, cities, sat_size=224, pano_size=(512,128), split="train"):
        super().__init__()
        self.split = split; self.pairs = []
        self.sat_tf = transforms.Compose([transforms.Resize((sat_size,sat_size)),transforms.ToTensor(),
             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        self.sat_aug = transforms.Compose([transforms.Resize((int(sat_size*1.1),int(sat_size*1.1))),
             transforms.RandomCrop((sat_size,sat_size)),transforms.RandomHorizontalFlip(),
             transforms.ColorJitter(0.2,0.15,0.1),transforms.ToTensor(),
             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        self.pano_tf = transforms.Compose([transforms.Resize(pano_size),transforms.ToTensor(),
             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        self.pano_aug = transforms.Compose([transforms.Resize(pano_size),transforms.RandomHorizontalFlip(),
             transforms.ColorJitter(0.2,0.15,0.1),transforms.ToTensor(),
             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

        for city in cities:
            city_dir = os.path.join(root, city, city)
            if not os.path.isdir(city_dir):
                city_dir = os.path.join(root, city)
            if not os.path.isdir(city_dir):
                print(f"  [WARN] {city} not found"); continue
            pano_dir = os.path.join(city_dir, "pano_images")
            sat_dir = os.path.join(city_dir, "sat_images")
            if not os.path.exists(pano_dir) or not os.path.exists(sat_dir):
                print(f"  [WARN] {city}: pano/sat dirs missing"); continue
            panos = sorted(glob.glob(os.path.join(pano_dir, "*.jpg")))
            for p in panos:
                name = os.path.splitext(os.path.basename(p))[0]
                s = os.path.join(sat_dir, name + ".jpg")
                if not os.path.exists(s): s = os.path.join(sat_dir, name + ".png")
                if os.path.exists(s):
                    self.pairs.append((p, s, city))
        print(f"  [CV-Cities {split}] {len(self.pairs)} pairs from {len(cities)} cities")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        pp, sp, city = self.pairs[idx]
        try: pano=Image.open(pp).convert("RGB"); sat=Image.open(sp).convert("RGB")
        except: pano=Image.new("RGB",(512,128),(128,128,128)); sat=Image.new("RGB",(224,224),(128,128,128))
        if self.split == "train":
            pano=self.pano_aug(pano); sat=self.sat_aug(sat)
        else:
            pano=self.pano_tf(pano); sat=self.sat_tf(sat)
        return {"query": pano, "gallery": sat, "city": city, "idx": idx}


# === EVALUATION ===
@torch.no_grad()
def evaluate_cities(model, root, cities, sat_size, pano_size, device):
    model.eval()
    all_results = {}
    for city in cities:
        ds = CVCitiesDataset(root, [city], sat_size, pano_size, "test")
        if len(ds) == 0: all_results[city] = {"R@1":0}; continue
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
        q_e, r_e = [], []
        for b in tqdm(loader, desc=f"Eval {city}", leave=False):
            q_e.append(model.extract_embedding(b["query"].to(device)).cpu())
            r_e.append(model.extract_embedding(b["gallery"].to(device)).cpu())
        qe = torch.cat(q_e,0).numpy(); re = torch.cat(r_e,0).numpy()
        N = len(qe); sim = qe @ re.T; ranks = np.argsort(-sim, axis=1)
        city_res = {}
        for k in [1,5,10]:
            city_res[f"R@{k}"] = sum(1 for i in range(N) if i in ranks[i,:k]) / N
        all_results[city] = city_res
        print(f"    {city}: R@1={city_res['R@1']:.2%}")
    # Average across cities
    avg = {f"R@{k}": np.mean([r.get(f"R@{k}",0) for r in all_results.values()]) for k in [1,5,10]}
    all_results["average"] = avg
    return all_results


# === TRAINING ===
def main():
    print("\n[1] Loading CV-Cities...")
    train_ds = CVCitiesDataset(CVCITIES_ROOT, TRAIN_CITIES, SAT_SIZE, PANO_SIZE, "train")
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

    log = {"dataset":"cv_cities","history":[]}; best_r1 = 0; gs = 0
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
            print(f"\n  Eval @ epoch {epoch+1} (cross-city)...")
            m = evaluate_cities(model, CVCITIES_ROOT, TEST_CITIES, SAT_SIZE, PANO_SIZE, DEVICE)
            entry["cross_city"] = m
            avg_r1 = m["average"]["R@1"]
            print(f"  Cross-city avg R@1 = {avg_r1:.2%}")
            if avg_r1 > best_r1:
                best_r1 = avg_r1
                torch.save({"epoch":epoch+1,"r1":avg_r1,"model_state_dict":model.state_dict()},
                           os.path.join(OUTPUT_DIR,"best_model_cv_cities.pth"))
                print(f"  ★ Best cross-city R@1: {avg_r1:.2%}")

        log["history"].append(entry)
        print(f"E{epoch+1} | Loss={el:.4f} | Acc={ea:.1%} | {time.time()-t0:.0f}s")

    log["best_r1"]=best_r1
    with open(os.path.join(OUTPUT_DIR,"results_cv_cities.json"),"w") as f: json.dump(log,f,indent=2)
    print(f"\n{'='*70}\n  Done! Best cross-city R@1 = {best_r1:.2%}\n{'='*70}")

if __name__=="__main__": main()
