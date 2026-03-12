"""
EXP7: GeoDINOL — DINOv2-Large Teacher (Simple Upgrade, Safest Bet)
=========================================================================
Teacher:  DINOv2 ViT-L/14 (300M params, 1024-dim) — same family, bigger
Student:  ConvNeXt-Tiny (same as baseline)
Novel:    Larger teacher + Feature Variance Regularization
Expected: 82-87% R@1 (most likely to beat baseline)
"""
import subprocess, importlib
def pip_install(pkg): subprocess.run(f"pip install -q {pkg}", shell=True, capture_output=True)
for p in ["timm","tqdm"]:
    try: importlib.import_module(p)
    except ImportError: pip_install(p)

import os, math, random, argparse, numpy as np

from typing import Dict, Any
from PIL import Image
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

class Config:
    DATA_ROOT="/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
    DRONE_DIR="drone-view"; SATELLITE_DIR="satellite-view"; OUTPUT_DIR="/kaggle/working"
    NUM_WORKERS=8; P=8; K=4; BATCH_SIZE=256; NUM_EPOCHS=120; LR=0.001; WARMUP_EPOCHS=5
    IMG_SIZE=224; NUM_CLASSES=120; EMBED_DIM=768; DROP_PATH_RATE=0.1
    TEMPERATURE=4.0; BASE_TEMPERATURE=4.0
    LAMBDA_TRIPLET=1.0; LAMBDA_CSC=0.5; LAMBDA_SELF_DIST=0.5
    LAMBDA_CROSS_DIST=0.3; LAMBDA_ALIGN=0.2; LAMBDA_FVR=0.1; MARGIN=0.3
    ALTITUDES=["150","200","250","300"]
    TRAIN_LOCS=list(range(1,121)); TEST_LOCS=list(range(121,201))
    USE_AMP=True; SEED=42

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark=True

# ============================================================================
# DATASET
# ============================================================================
class SUES200Dataset(Dataset):
    def __init__(self, root, mode="train", altitudes=None, transform=None, train_locs=None, test_locs=None):
        self.root=root; self.mode=mode; self.altitudes=altitudes or Config.ALTITUDES; self.transform=transform
        self.drone_dir=os.path.join(root,Config.DRONE_DIR); self.satellite_dir=os.path.join(root,Config.SATELLITE_DIR)
        train_locs=train_locs or Config.TRAIN_LOCS; test_locs=test_locs or Config.TEST_LOCS
        loc_ids=train_locs if mode=="train" else test_locs
        self.locations=[f"{l:04d}" for l in loc_ids]; self.location_to_idx={l:i for i,l in enumerate(self.locations)}
        self.samples=[]; self.drone_by_location=defaultdict(list)
        for loc in self.locations:
            li=self.location_to_idx[loc]; sp=os.path.join(self.satellite_dir,loc,"0.png")
            if not os.path.exists(sp): continue
            for alt in self.altitudes:
                ad=os.path.join(self.drone_dir,loc,alt)
                if not os.path.isdir(ad): continue
                for img in sorted(os.listdir(ad)):
                    if img.endswith(('.png','.jpg','.jpeg')):
                        self.samples.append((os.path.join(ad,img),sp,li,alt))
                        self.drone_by_location[li].append(len(self.samples)-1)
        print(f"[{mode}] {len(self.samples)} samples, {len(self.locations)} locs")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        dp,sp,li,alt=self.samples[idx]
        d=Image.open(dp).convert('RGB'); s=Image.open(sp).convert('RGB')
        if self.transform: d=self.transform(d); s=self.transform(s)
        return {'drone':d,'satellite':s,'label':li,'altitude':int(alt)}

class PKSampler:
    def __init__(self, ds, p=8, k=4): self.ds=ds; self.p=p; self.k=k; self.locs=list(ds.drone_by_location.keys())
    def __iter__(self):
        locs=self.locs.copy(); random.shuffle(locs); batch=[]
        for l in locs:
            idx=self.ds.drone_by_location[l]
            if len(idx)<self.k: idx=idx*(self.k//len(idx)+1)
            batch.extend(random.sample(idx,self.k))
            if len(batch)>=self.p*self.k: yield batch[:self.p*self.k]; batch=batch[self.p*self.k:]
    def __len__(self): return len(self.locs)//self.p

def get_transforms(mode="train"):
    if mode=="train":
        return T.Compose([T.Resize((Config.IMG_SIZE,Config.IMG_SIZE)),T.RandomHorizontalFlip(0.5),
            T.RandomResizedCrop(Config.IMG_SIZE,scale=(0.8,1.0)),T.ColorJitter(0.2,0.2,0.2),
            T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return T.Compose([T.Resize((Config.IMG_SIZE,Config.IMG_SIZE)),T.ToTensor(),
                      T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# ============================================================================
# CONVNEXT-TINY BACKBONE
# ============================================================================
class LayerNorm(nn.Module):
    def __init__(self, ns, eps=1e-6, df="channels_last"):
        super().__init__(); self.weight=nn.Parameter(torch.ones(ns)); self.bias=nn.Parameter(torch.zeros(ns))
        self.eps=eps; self.df=df; self.ns=(ns,)
    def forward(self, x):
        if self.df=="channels_last": return F.layer_norm(x,self.ns,self.weight,self.bias,self.eps)
        u=x.mean(1,keepdim=True); s=(x-u).pow(2).mean(1,keepdim=True)
        x=(x-u)/torch.sqrt(s+self.eps); return self.weight[:,None,None]*x+self.bias[:,None,None]

def drop_path(x, dp=0., tr=False):
    if dp==0. or not tr: return x
    kp=1-dp; sh=(x.shape[0],)+(1,)*(x.ndim-1)
    rt=kp+torch.rand(sh,dtype=x.dtype,device=x.device); rt.floor_(); return x.div(kp)*rt
class DropPath(nn.Module):
    def __init__(self, dp=None): super().__init__(); self.dp=dp
    def forward(self, x): return drop_path(x,self.dp,self.training)

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, dpr=0., lsi=1e-6):
        super().__init__()
        self.dwconv=nn.Conv2d(dim,dim,7,padding=3,groups=dim); self.norm=LayerNorm(dim)
        self.pwconv1=nn.Linear(dim,4*dim); self.act=nn.GELU(); self.pwconv2=nn.Linear(4*dim,dim)
        self.gamma=nn.Parameter(lsi*torch.ones(dim)) if lsi>0 else None
        self.drop_path=DropPath(dpr) if dpr>0 else nn.Identity()
    def forward(self, x):
        sc=x; x=self.dwconv(x); x=x.permute(0,2,3,1); x=self.norm(x)
        x=self.pwconv1(x); x=self.act(x); x=self.pwconv2(x)
        if self.gamma is not None: x=self.gamma*x
        x=x.permute(0,3,1,2); return sc+self.drop_path(x)

class ConvNeXtTiny(nn.Module):
    def __init__(self, in_c=3, depths=[3,3,9,3], dims=[96,192,384,768], dpr=0., lsi=1e-6):
        super().__init__(); self.dims=dims; self.downsample_layers=nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(nn.Conv2d(in_c,dims[0],4,stride=4),LayerNorm(dims[0],data_format="channels_first")))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(LayerNorm(dims[i],data_format="channels_first"),nn.Conv2d(dims[i],dims[i+1],2,stride=2)))
        dp=[x.item() for x in torch.linspace(0,dpr,sum(depths))]; cur=0; self.stages=nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[ConvNeXtBlock(dims[i],dp[cur+j],lsi) for j in range(depths[i])])); cur+=depths[i]
        self.norm=nn.LayerNorm(dims[-1]); self.apply(self._iw)
    def _iw(self,m):
        if isinstance(m,(nn.Conv2d,nn.Linear)):
            nn.init.trunc_normal_(m.weight,std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias,0)
    def forward_features(self,x):
        so=[]
        for i in range(4): x=self.downsample_layers[i](x); x=self.stages[i](x); so.append(x)
        return self.norm(x.mean([-2,-1])), so
    def forward(self,x): return self.forward_features(x)

def load_convnext_pretrained(m):
    try:
        ckpt=torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth",map_location="cpu",check_hash=True)
        m.load_state_dict({k:v for k,v in ckpt["model"].items() if not k.startswith('head')},strict=False)
        print("Loaded ConvNeXt-Tiny pretrained")
    except Exception as e: print(f"Pretrained load failed: {e}")
    return m

# ============================================================================
# STUDENT
# ============================================================================
class ClassificationHead(nn.Module):
    def __init__(self, d, nc, hd=512):
        super().__init__(); self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Linear(d,hd),nn.BatchNorm1d(hd),nn.ReLU(True),nn.Dropout(0.5),nn.Linear(hd,nc))
    def forward(self,x): return self.fc(self.pool(x).flatten(1))

class MobileGeoStudent(nn.Module):
    def __init__(self, nc, ed=768, dpr=0.1):
        super().__init__()
        self.backbone=load_convnext_pretrained(ConvNeXtTiny(dpr=dpr))
        self.dims=[96,192,384,768]
        self.aux_heads=nn.ModuleList([ClassificationHead(d,nc) for d in self.dims])
        self.bottleneck=nn.Sequential(nn.Linear(768,ed),nn.BatchNorm1d(ed),nn.ReLU(True))
        self.classifier=nn.Linear(ed,nc)
    def forward(self, x, return_all=False):
        ff, so = self.backbone(x)
        sl=[h(f) for h,f in zip(self.aux_heads,so)]
        emb=self.bottleneck(ff); en=F.normalize(emb,p=2,dim=1); logits=self.classifier(emb)
        if return_all:
            return {'embedding':emb,'embedding_normed':en,'logits':logits,
                    'stage_logits':sl,'stage_features':so,'final_feature':ff}
        return en, logits

# ============================================================================
# DINOV2-LARGE TEACHER
# ============================================================================
class DINOv2LargeTeacher(nn.Module):
    """DINOv2 ViT-L/14 — 300M params, 1024-dim (3.5× larger than ViT-B baseline)"""
    def __init__(self, ntb=2):
        super().__init__()
        print("Loading DINOv2 ViT-L/14...")
        self.model=torch.hub.load('facebookresearch/dinov2','dinov2_vitl14')
        self.num_channels=1024
        for p in self.model.parameters(): p.requires_grad=False
        for blk in self.model.blocks[-ntb:]:
            for p in blk.parameters(): p.requires_grad=True
        t=sum(p.numel() for p in self.model.parameters()); tr=sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"DINOv2-Large: {t/1e6:.0f}M total, {tr/1e6:.1f}M trainable")
    @torch.no_grad()
    def forward(self,x):
        x=self.model.prepare_tokens_with_masks(x)
        for blk in self.model.blocks: x=blk(x)
        return self.model.norm(x)[:,0]

# ============================================================================
# LOSSES
# ============================================================================
class TripletLoss(nn.Module):
    def __init__(self, m=0.3): super().__init__(); self.m=m
    def forward(self, e, l):
        d=torch.cdist(e,e,p=2); l=l.view(-1,1); mp=l.eq(l.T).float(); mn=l.ne(l.T).float()
        return F.relu((d*mp).max(1)[0]-(d*mn+mp*1e9).min(1)[0]+self.m).mean()

class SymmetricInfoNCELoss(nn.Module):
    def __init__(self, t=0.07): super().__init__(); self.t=t
    def forward(self,df,sf,l):
        df=F.normalize(df,dim=1); sf=F.normalize(sf,dim=1)
        sim=df@sf.T/self.t; l=l.view(-1,1); pm=l.eq(l.T).float()
        ld=-torch.log((F.softmax(sim,1)*pm).sum(1)/pm.sum(1).clamp(min=1)).mean()
        ls=-torch.log((F.softmax(sim.T,1)*pm).sum(1)/pm.sum(1).clamp(min=1)).mean()
        return 0.5*(ld+ls)

class SelfDistillationLoss(nn.Module):
    def __init__(self, t=4.0, w=[0.1,0.2,0.3,0.4]): super().__init__(); self.t=t; self.w=w
    def forward(self, sl):
        loss=0.; fl=sl[-1]
        for i in range(len(sl)-1):
            loss+=self.w[i]*(self.t**2)*F.kl_div(F.log_softmax(fl/self.t,1),F.softmax(sl[i]/self.t,1),reduction='batchmean')
        return loss

class UAPALoss(nn.Module):
    def __init__(self, bt=4.0): super().__init__(); self.T0=bt
    def forward(self,dl,sl):
        pd=F.softmax(dl,1); ps=F.softmax(sl,1)
        Ud=-(pd*torch.log(pd+1e-8)).sum(1).mean(); Us=-(ps*torch.log(ps+1e-8)).sum(1).mean()
        T=self.T0*(1+torch.sigmoid(Ud-Us))
        return (T**2)*F.kl_div(F.log_softmax(dl/T,1),F.softmax(sl/T,1),reduction='batchmean')

class CrossDistillationLoss(nn.Module):
    def __init__(self, sd=768, td=1024):
        super().__init__()
        self.proj=nn.Linear(sd,td) if sd!=td else nn.Identity()
    def forward(self,sf,tf):
        s=F.normalize(self.proj(sf),dim=1); t=F.normalize(tf,dim=1)
        return F.mse_loss(s,t)+(1-F.cosine_similarity(s,t).mean())

class FeatureVarianceReg(nn.Module):
    """Prevent student feature collapse by matching teacher variance"""
    def forward(self,sf,tf):
        with torch.no_grad(): tv=tf.var(0).mean()
        return F.mse_loss(sf.var(0).mean(), tv)

class MobileGeoLoss(nn.Module):
    def __init__(self, nc, cfg=Config):
        super().__init__(); self.cfg=cfg; self.ce=nn.CrossEntropyLoss()
        self.triplet=TripletLoss(cfg.MARGIN); self.csc=SymmetricInfoNCELoss()
        self.sd=SelfDistillationLoss(cfg.TEMPERATURE); self.uapa=UAPALoss(cfg.BASE_TEMPERATURE)
        self.cd=CrossDistillationLoss(768,1024); self.fvr=FeatureVarianceReg()
    def forward(self,do,so,l,td=None,ts=None):
        losses={}
        ce=sum(0.25*self.ce(lg,l) for lg in do['stage_logits'])+self.ce(do['logits'],l)
        ce+=sum(0.25*self.ce(lg,l) for lg in so['stage_logits'])+self.ce(so['logits'],l)
        losses['ce']=ce
        losses['triplet']=self.cfg.LAMBDA_TRIPLET*(self.triplet(do['embedding_normed'],l)+self.triplet(so['embedding_normed'],l))
        losses['csc']=self.cfg.LAMBDA_CSC*self.csc(do['embedding_normed'],so['embedding_normed'],l)
        losses['self_dist']=self.cfg.LAMBDA_SELF_DIST*(self.sd(do['stage_logits'])+self.sd(so['stage_logits']))
        losses['uapa']=self.cfg.LAMBDA_ALIGN*self.uapa(do['logits'],so['logits'])
        if td is not None:
            losses['cross_dist']=self.cfg.LAMBDA_CROSS_DIST*(self.cd(do['final_feature'],td)+self.cd(so['final_feature'],ts))
            losses['fvr']=self.cfg.LAMBDA_FVR*(self.fvr(do['final_feature'],td)+self.fvr(so['final_feature'],ts))
        total=sum(losses.values()); losses['total']=total; return total, losses

# ============================================================================
# EVALUATION
# ============================================================================
# ============================================================================
# TRAINING
# ============================================================================

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

class WarmupCosineScheduler:
    def __init__(self,opt,we,te,mlr=1e-6):
        self.opt=opt;self.we=we;self.te=te;self.mlr=mlr;self.blr=opt.param_groups[0]['lr']
    def step(self,e):
        if e<self.we: lr=self.blr*(e+1)/self.we
        else: lr=self.mlr+0.5*(self.blr-self.mlr)*(1+math.cos(math.pi*(e-self.we)/(self.te-self.we)))
        for pg in self.opt.param_groups: pg['lr']=lr
        return lr

def train_one_epoch(model,teacher,tl,crit,opt,scaler,dev,epoch,cfg=Config):
    model.train()
    if teacher: teacher.eval()
    tl_sum=0; ld_sum=defaultdict(float)
    for bi,b in enumerate(tl):
        di=b['drone'].to(dev); si=b['satellite'].to(dev); l=b['label'].to(dev); opt.zero_grad()
        with torch.amp.autocast('cuda',enabled=cfg.USE_AMP):
            do=model(di,return_all=True); so=model(si,return_all=True)
            td=ts=None
            if teacher:
                with torch.no_grad(): td=teacher(di); ts=teacher(si)
            loss,ld=crit(do,so,l,td,ts)
        if cfg.USE_AMP: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else: loss.backward(); opt.step()
        tl_sum+=loss.item()
        for k,v in ld.items(): ld_sum[k]+=v.item() if torch.is_tensor(v) else v
        if bi%20==0: print(f"  Batch {bi}/{len(tl)}, Loss:{loss.item():.4f}")
    if len(tl)==0: return 0.,{}
    return tl_sum/len(tl), {k:v/len(tl) for k,v in ld_sum.items()}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--epochs",type=int,default=Config.NUM_EPOCHS)
    parser.add_argument("--batch_size",type=int,default=Config.BATCH_SIZE)
    parser.add_argument("--data_root",type=str,default=Config.DATA_ROOT)
    parser.add_argument("--test",action="store_true"); args,_=parser.parse_known_args()
    Config.NUM_EPOCHS=args.epochs; Config.BATCH_SIZE=args.batch_size; Config.DATA_ROOT=args.data_root
    if args.test: Config.NUM_EPOCHS=1;Config.NUM_WORKERS=0;Config.BATCH_SIZE=8;Config.P=2
    Config.K=max(2,Config.BATCH_SIZE//Config.P)
    print("="*60+"\nEXP7: GeoDINOL — DINOv2-Large Teacher\n"+"="*60)
    set_seed(Config.SEED); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trd=SUES200Dataset(Config.DATA_ROOT,"train",transform=get_transforms("train"))
    ted=SUES200Dataset(Config.DATA_ROOT,"test",transform=get_transforms("test"))
    nc=len(Config.TRAIN_LOCS)
    trl=DataLoader(trd,batch_sampler=PKSampler(trd,Config.P,Config.K),num_workers=Config.NUM_WORKERS,pin_memory=True)
    model=MobileGeoStudent(nc,Config.EMBED_DIM).to(dev)
    try: teacher=DINOv2LargeTeacher(2).to(dev)
    except Exception as e: print(f"Teacher failed:{e}"); teacher=None
    crit=MobileGeoLoss(nc).to(dev)
    opt=torch.optim.SGD(list(model.parameters())+list(crit.parameters()),lr=Config.LR,momentum=0.9,weight_decay=5e-4)
    sched=WarmupCosineScheduler(opt,Config.WARMUP_EPOCHS,Config.NUM_EPOCHS)
    scaler=torch.amp.GradScaler('cuda',enabled=Config.USE_AMP); best=0.
    for ep in range(Config.NUM_EPOCHS):
        lr=sched.step(ep); print(f"\n{'='*40}\nEp {ep+1}/{Config.NUM_EPOCHS} LR:{lr:.6f}\n{'='*40}")
        al,ld=train_one_epoch(model,teacher,trl,crit,opt,scaler,dev,ep)
        print(f"Ep {ep+1} Loss:{al:.4f}"); [print(f"  {k}:{v:.4f}") for k,v in ld.items()]
        if (ep+1)%5==0 or ep==Config.NUM_EPOCHS-1:
            r,ap,_=evaluate(model,ted,dev,Config)
            print(f"  R@1:{r['R@1']:.2f}% R@5:{r['R@5']:.2f}% R@10:{r['R@10']:.2f}% mAP:{ap:.2f}%")
            if r['R@1']>best:
                best=r['R@1']; torch.save({'epoch':ep,'model':model.state_dict(),'r1':r['R@1'],'ap':ap},
                    os.path.join(Config.OUTPUT_DIR,'best_model_exp7.pth')); print("  ★ Best!")
    print(f"\n{'='*60}\nDone! Best R@1:{best:.2f}%\n{'='*60}")



if __name__=="__main__": main()
