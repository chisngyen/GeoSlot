# =============================================================================
# PHASE 2: GeoSlot — Train on University-1652 (Main Benchmark #1)
# Target: Drone→Sat R@1 ≥ 97% | SOTA: 96.88% (OG-Sample4Geo, Jan 2025)
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
UNI1652_ROOT = "/kaggle/input/datasets/chinguyeen/university-1652/University-1652"
OUTPUT_DIR   = "/kaggle/working"
RESUME_FROM  = None  # e.g. "/kaggle/working/best_model_cvusa.pth"

IMG_SIZE     = 384;  BATCH_SIZE = 32;  EPOCHS = 60
LR_BB = 1e-5;  LR_HEAD = 1e-4;  WARMUP = 3;  FREEZE_BB = 5
EVAL_FREQ = 5;  SAVE_FREQ = 10;  S2_EPOCH = 20;  S3_EPOCH = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("  PHASE 2: GeoSlot — University-1652")
print(f"  Target: Drone→Sat R@1 ≥ 97% | SOTA: 96.88%")
print(f"  Device: {DEVICE} | Image: {IMG_SIZE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")
print("=" * 70)


# === DATASET ===
class University1652Dataset(Dataset):
    def __init__(self, root, split="train", img_size=384):
        super().__init__()
        self.split = split
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.tf_aug = transforms.Compose([
            transforms.Resize((int(img_size*1.1), int(img_size*1.1))),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.15, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.pairs = []; self.query_imgs = []; self.query_labels = []
        self.gallery_imgs = []; self.gallery_labels = []
        if split == "train": self._load_train(root)
        else: self._load_test(root)

    def _find_imgs(self, d):
        return sorted(glob.glob(os.path.join(d, "**", "*.jp*g"), recursive=True) +
                       glob.glob(os.path.join(d, "**", "*.png"), recursive=True))

    def _load_train(self, root):
        drone_dir = os.path.join(root, "train", "drone")
        sat_dir = os.path.join(root, "train", "satellite")
        if not os.path.exists(drone_dir): print(f"  [WARN] {drone_dir} not found"); return
        for cls in sorted(os.listdir(drone_dir)):
            dc = os.path.join(drone_dir, cls); sc = os.path.join(sat_dir, cls)
            if not os.path.isdir(dc) or not os.path.isdir(sc): continue
            d_imgs = self._find_imgs(dc); s_imgs = self._find_imgs(sc)
            if not d_imgs or not s_imgs: continue
            for d in d_imgs:
                self.pairs.append((d, s_imgs[0], cls))
        print(f"  [Uni1652 train] {len(self.pairs)} pairs")

    def _load_test(self, root):
        test = os.path.join(root, "test")
        for qn in ["query_drone", "drone"]:
            qd = os.path.join(test, qn)
            if os.path.exists(qd): break
        for gn in ["gallery_satellite", "satellite"]:
            gd = os.path.join(test, gn)
            if os.path.exists(gd): break
        for d in [qd, gd]:
            if not os.path.exists(d): continue
            is_q = "query" in d or "drone" in d
            for cls in sorted(os.listdir(d)):
                cd = os.path.join(d, cls)
                if not os.path.isdir(cd): continue
                for ip in self._find_imgs(cd):
                    if is_q: self.query_imgs.append(ip); self.query_labels.append(cls)
                    else: self.gallery_imgs.append(ip); self.gallery_labels.append(cls)
        print(f"  [Uni1652 test] {len(self.query_imgs)} queries, {len(self.gallery_imgs)} gallery")

    def __len__(self):
        return len(self.pairs) if self.split == "train" else len(self.query_imgs)

    def __getitem__(self, idx):
        if self.split == "train":
            dp, sp, cls = self.pairs[idx]
            try: d=Image.open(dp).convert("RGB"); s=Image.open(sp).convert("RGB")
            except: d=s=Image.new("RGB",(384,384),(128,128,128))
            return {"query": self.tf_aug(d), "gallery": self.tf_aug(s), "class_id": cls}
        else:
            try: img = Image.open(self.query_imgs[idx]).convert("RGB")
            except: img = Image.new("RGB",(384,384),(128,128,128))
            return {"image": self.tf(img), "label": self.query_labels[idx]}


# === EVALUATION ===
@torch.no_grad()
def evaluate(model, root, img_size, device):
    model.eval()
    tf = transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor(),
         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    def extract(d):
        embs, labs = [], []
        if not os.path.exists(d): return np.array([]), np.array([])
        for cls in sorted(os.listdir(d)):
            cd = os.path.join(d, cls)
            if not os.path.isdir(cd): continue
            for ip in sorted(glob.glob(os.path.join(cd,"**","*.jp*g"),recursive=True)):
                try:
                    img = tf(Image.open(ip).convert("RGB")).unsqueeze(0).to(device)
                    embs.append(model.extract_embedding(img).cpu().numpy()[0]); labs.append(cls)
                except: pass
        return np.array(embs), np.array(labs)

    test = os.path.join(root, "test")
    for qn in ["query_drone","drone"]:
        qd=os.path.join(test,qn)
        if os.path.exists(qd): break
    for gn in ["gallery_satellite","satellite"]:
        gd=os.path.join(test,gn)
        if os.path.exists(gd): break

    print("    Extracting queries..."); qe, ql = extract(qd)
    print(f"    Queries: {len(qe)}")
    print("    Extracting gallery..."); ge, gl = extract(gd)
    print(f"    Gallery: {len(ge)}")

    if len(qe)==0 or len(ge)==0: return {"R@1":0,"R@5":0,"R@10":0,"AP":0}
    sim = qe @ ge.T; ranks = np.argsort(-sim, axis=1)
    results = {}
    for k in [1,5,10]:
        results[f"R@{k}"] = sum(1 for i in range(len(ql)) if ql[i] in gl[ranks[i,:k]]) / len(ql)
    # AP
    ap = 0.0
    for i in range(len(ql)):
        rel = (gl[ranks[i]] == ql[i]).astype(float)
        if rel.sum() == 0: continue
        prec = np.cumsum(rel) / (np.arange(len(rel))+1)
        ap += (prec * rel).sum() / rel.sum()
    results["AP"] = ap / len(ql)
    return results


# === TRAINING ===
def main():
    print("\n[1/5] Loading University-1652...")
    train_ds = University1652Dataset(UNI1652_ROOT, "train", IMG_SIZE)
    if len(train_ds) == 0:
        print("[ERROR] No data! Check path."); return
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)

    print("\n[2/5] Building model...")
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

    log = {"dataset":"university1652","history":[]}; best_r1 = 0; gs = 0
    print(f"\n[3/5] Training ({EPOCHS} epochs)...\n")

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
            print(f"\n  Eval @ epoch {epoch+1}...")
            m = evaluate(model, UNI1652_ROOT, IMG_SIZE, DEVICE)
            entry.update(m); r1=m.get("R@1",0)
            print(f"  R@1={r1:.2%} R@5={m.get('R@5',0):.2%} AP={m.get('AP',0):.2%}")
            if r1 > best_r1:
                best_r1 = r1
                torch.save({"epoch":epoch+1,"r1":r1,"model_state_dict":model.state_dict()},
                           os.path.join(OUTPUT_DIR,"best_model_uni1652.pth"))
                print(f"  ★ Best R@1: {r1:.2%}")
        log["history"].append(entry)
        print(f"E{epoch+1} | Loss={el:.4f} | Acc={ea:.1%} | {time.time()-t0:.0f}s")
        if (epoch+1)%SAVE_FREQ==0:
            torch.save({"epoch":epoch+1,"model_state_dict":model.state_dict()},
                       os.path.join(OUTPUT_DIR,f"ckpt_uni1652_ep{epoch+1}.pth"))

    log["best_r1"]=best_r1
    with open(os.path.join(OUTPUT_DIR,"results_uni1652.json"),"w") as f: json.dump(log,f,indent=2)
    print(f"\n{'='*70}\n  Done! Best Drone→Sat R@1 = {best_r1:.2%}\n{'='*70}")

if __name__=="__main__": main()
