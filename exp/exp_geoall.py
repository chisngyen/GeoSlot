#!/usr/bin/env python3
"""
GeoAll: Unified Best-of-Everything for Cross-View Drone Geo-Localization
=========================================================================
Built on: baseline.py (MobileGeo — 82.35% R@1)
Novel contributions (combined from proven experiments):
  1. SAM optimizer + EMA + Gradient Centralization (from GeoSAM)
  2. Local Branch Classifiers (gated residual) (from GeoAGEN — 69.98%)
  3. Fuzzy PID Controller (from GeoAGEN) — but DELAYED past warmup

Design principles:
  - Start from baseline architecture exactly
  - Local branches are ADDITIVE (gate=0 init) so NCE sees pretrained features
  - PID delayed until after warmup so it gets meaningful error signals
  - Keep GradScaler OFF (SAM incompatible)
"""
import subprocess, os
def pip_install(pkg): subprocess.run(f"pip install -q {pkg}", shell=True, capture_output=True)
print("[1/2] Installing packages...")
for p in ["timm","tqdm"]:
    try: __import__(p)
    except: pip_install(p)
print("[2/2] Setup complete!")
import math, random, argparse
import numpy as np
from PIL import Image
from collections import defaultdict
from typing import Dict, Any
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import torchvision.transforms as T
print("[OK] All imports loaded!")

class Config:
    DATA_ROOT="/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"; DRONE_DIR="drone-view"
    SATELLITE_DIR="satellite-view"; OUTPUT_DIR="/kaggle/working"
    NUM_WORKERS=8; P=8; K=4; BATCH_SIZE=256; NUM_EPOCHS=120; LR=0.001; WARMUP_EPOCHS=5
    IMG_SIZE=224; NUM_CLASSES=120; EMBED_DIM=768; DROP_PATH_RATE=0.1
    TEMPERATURE=4.0; BASE_TEMPERATURE=4.0
    LAMBDA_TRIPLET=1.0; LAMBDA_CSC=0.5; LAMBDA_SELF_DIST=0.5
    LAMBDA_CROSS_DIST=0.3; LAMBDA_ALIGN=0.2; MARGIN=0.3
    ALTITUDES=["150","200","250","300"]
    TRAIN_LOCS=list(range(1,121)); TEST_LOCS=list(range(121,201))
    USE_AMP=True; SEED=42; SAM_RHO=0.05; EMA_DECAY=0.9999
    NUM_LOCAL=4  # number of local branch strips

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); torch.backends.cudnn.benchmark=True

# === DATASET ===
class SUES200Dataset(Dataset):
    def __init__(self,root,mode="train",altitudes=None,transform=None,train_locs=None,test_locs=None):
        self.root=root; self.altitudes=altitudes or Config.ALTITUDES; self.transform=transform
        self.drone_dir=os.path.join(root,Config.DRONE_DIR); self.satellite_dir=os.path.join(root,Config.SATELLITE_DIR)
        if train_locs is None: train_locs=Config.TRAIN_LOCS
        if test_locs is None: test_locs=Config.TEST_LOCS
        locs=train_locs if mode=="train" else test_locs
        self.locations=[f"{l:04d}" for l in locs]; self.location_to_idx={l:i for i,l in enumerate(self.locations)}
        self.samples=[]; self.drone_by_location=defaultdict(list)
        for loc in self.locations:
            li=self.location_to_idx[loc]; sp=os.path.join(self.satellite_dir,loc,"0.png")
            if not os.path.exists(sp): continue
            for alt in self.altitudes:
                ad=os.path.join(self.drone_dir,loc,alt)
                if not os.path.isdir(ad): continue
                for n in sorted(os.listdir(ad)):
                    if n.endswith(('.png','.jpg','.jpeg')):
                        self.samples.append((os.path.join(ad,n),sp,li,alt)); self.drone_by_location[li].append(len(self.samples)-1)
        print(f"[{mode}] {len(self.samples)} samples, {len(self.locations)} locs")
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        dp,sp,li,alt=self.samples[idx]; d=Image.open(dp).convert('RGB'); s=Image.open(sp).convert('RGB')
        if self.transform: d=self.transform(d); s=self.transform(s)
        return {'drone':d,'satellite':s,'label':li,'altitude':int(alt)}

class PKSampler:
    def __init__(self,d,p=8,k=4): self.d=d; self.p=p; self.k=k; self.locs=list(d.drone_by_location.keys())
    def __iter__(self):
        locs=self.locs.copy(); random.shuffle(locs); batch=[]
        for loc in locs:
            idx=self.d.drone_by_location[loc]
            if len(idx)<self.k: idx=idx*(self.k//len(idx)+1)
            batch.extend(random.sample(idx,self.k))
            if len(batch)>=self.p*self.k: yield batch[:self.p*self.k]; batch=batch[self.p*self.k:]
    def __len__(self): return len(self.locs)//self.p

def get_transforms(mode="train"):
    if mode=="train": return T.Compose([T.Resize((Config.IMG_SIZE,)*2),T.RandomHorizontalFlip(0.5),
        T.RandomResizedCrop(Config.IMG_SIZE,scale=(0.8,1.0)),T.ColorJitter(0.2,0.2,0.2),
        T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return T.Compose([T.Resize((Config.IMG_SIZE,)*2),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# === BACKBONE ===
class LayerNorm(nn.Module):
    def __init__(self,n,eps=1e-6,data_format="channels_last"):
        super().__init__(); self.weight=nn.Parameter(torch.ones(n)); self.bias=nn.Parameter(torch.zeros(n))
        self.eps=eps; self.df=data_format; self.ns=(n,)
    def forward(self,x):
        if self.df=="channels_last": return F.layer_norm(x,self.ns,self.weight,self.bias,self.eps)
        u=x.mean(1,keepdim=True); s=(x-u).pow(2).mean(1,keepdim=True)
        return self.weight[:,None,None]*((x-u)/torch.sqrt(s+self.eps))+self.bias[:,None,None]

def drop_path(x,p=0.,training=False):
    if p==0. or not training: return x
    kp=1-p; sh=(x.shape[0],)+(1,)*(x.ndim-1); rt=kp+torch.rand(sh,dtype=x.dtype,device=x.device); rt.floor_(); return x.div(kp)*rt

class DropPath(nn.Module):
    def __init__(self,p=None): super().__init__(); self.p=p
    def forward(self,x): return drop_path(x,self.p,self.training)

class ConvNeXtBlock(nn.Module):
    def __init__(self,dim,dpr=0.,lsi=1e-6):
        super().__init__()
        self.dw=nn.Conv2d(dim,dim,7,padding=3,groups=dim); self.norm=LayerNorm(dim)
        self.pw1=nn.Linear(dim,4*dim); self.act=nn.GELU(); self.pw2=nn.Linear(4*dim,dim)
        self.gamma=nn.Parameter(lsi*torch.ones(dim)) if lsi>0 else None
        self.dp=DropPath(dpr) if dpr>0 else nn.Identity()
    def forward(self,x):
        sc=x; x=self.dw(x); x=x.permute(0,2,3,1); x=self.norm(x); x=self.pw1(x); x=self.act(x); x=self.pw2(x)
        if self.gamma is not None: x=self.gamma*x
        return sc+self.dp(x.permute(0,3,2,1).permute(0,1,3,2))

class ConvNeXtTiny(nn.Module):
    def __init__(self,depths=[3,3,9,3],dims=[96,192,384,768],dpr=0.):
        super().__init__(); self.dims=dims; self.dl=nn.ModuleList()
        self.dl.append(nn.Sequential(nn.Conv2d(3,dims[0],4,4),LayerNorm(dims[0],data_format="channels_first")))
        for i in range(3): self.dl.append(nn.Sequential(LayerNorm(dims[i],data_format="channels_first"),nn.Conv2d(dims[i],dims[i+1],2,2)))
        rates=[x.item() for x in torch.linspace(0,dpr,sum(depths))]; cur=0; self.stages=nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[ConvNeXtBlock(dims[i],rates[cur+j]) for j in range(depths[i])])); cur+=depths[i]
        self.norm=nn.LayerNorm(dims[-1],eps=1e-6); self.apply(self._iw)
    def _iw(self,m):
        if isinstance(m,(nn.Conv2d,nn.Linear)): nn.init.trunc_normal_(m.weight,std=0.02); m.bias is not None and nn.init.constant_(m.bias,0)
    def forward(self,x):
        outs=[]
        for i in range(4): x=self.dl[i](x); x=self.stages[i](x); outs.append(x)
        return self.norm(x.mean([-2,-1])), outs

def load_convnext_pretrained(m):
    try:
        ck=torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth",map_location="cpu",check_hash=True)
        m.load_state_dict({k:v for k,v in ck["model"].items() if not k.startswith('head')},strict=False); print("Loaded ConvNeXt-Tiny pretrained (ImageNet-22K)")
    except Exception as e: print(f"Could not load pretrained: {e}")
    return m

class ClassificationHead(nn.Module):
    def __init__(self,d,nc,h=512):
        super().__init__(); self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Linear(d,h),nn.BatchNorm1d(h),nn.ReLU(True),nn.Dropout(0.5),nn.Linear(h,nc))
    def forward(self,x): return self.fc(self.pool(x).flatten(1))

# === NOVEL: LOCAL BRANCH CLASSIFIERS (from GeoAGEN) ===
class LocalBranchClassifiers(nn.Module):
    """Horizontal strip pooling: split feature map into N strips → classify each.
    Returns aggregated local feature [B, 768] for gated residual addition.
    """
    def __init__(self, in_dim=768, num_classes=120, num_branches=4):
        super().__init__()
        self.num_branches = num_branches
        self.branch_pools = nn.ModuleList([nn.AdaptiveAvgPool2d((1, None)) for _ in range(num_branches)])
        self.branch_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, in_dim//2), nn.BatchNorm1d(in_dim//2), nn.ReLU(True),
                          nn.Linear(in_dim//2, num_classes)) for _ in range(num_branches)])
        # Project back to in_dim for residual addition
        self.local_proj = nn.Sequential(nn.Linear(in_dim, in_dim), nn.BatchNorm1d(in_dim), nn.ReLU(True))

    def forward(self, feat_map):
        B, C, H, W = feat_map.shape
        strips = torch.chunk(feat_map, self.num_branches, dim=2)  # split along H
        branch_logits = []
        branch_feats = []
        for i, (pool, head) in enumerate(zip(self.branch_pools, self.branch_heads)):
            if i < len(strips):
                s = strips[i]
            else:
                s = feat_map
            f = pool(s).flatten(1)            # [B, C] (pool across W; H=1 after strips)
            # Handle case where strip is too small for pool
            if f.shape[-1] != C:
                f = feat_map.mean([-2,-1])
            branch_logits.append(head(f))
            branch_feats.append(f)
        local_feat = torch.mean(torch.stack(branch_feats, dim=1), dim=1)  # [B, C]
        return self.local_proj(local_feat), branch_logits


# === NOVEL: FUZZY PID CONTROLLER ===
class FuzzyPIDController:
    def __init__(self, loss_names, kp=0.1, ki=0.01, kd=0.05, decay=0.95):
        self.names=loss_names; self.kp=kp; self.ki=ki; self.kd=kd; self.decay=decay
        self.weights={n:1.0 for n in loss_names}; self.targets={n:None for n in loss_names}
        self.prev={n:0. for n in loss_names}; self.integral={n:0. for n in loss_names}
    def update(self, losses):
        for n in self.names:
            v=losses.get(n,0.)
            if self.targets[n] is None: self.targets[n]=v; continue
            err=v-self.targets[n]; self.integral[n]=0.9*self.integral[n]+err; der=err-self.prev[n]
            adj=self.kp*err+self.ki*self.integral[n]+self.kd*der
            if abs(err)<0.1: adj*=0.5
            elif abs(err)>1.0: adj*=1.5
            self.weights[n]=max(0.5, min(2.0, self.weights[n]+adj))
            self.prev[n]=err; self.targets[n]*=self.decay
    def get_weights(self): return dict(self.weights)


# === STUDENT: BASELINE + LOCAL BRANCHES (gated) ===
class GeoAllStudent(nn.Module):
    """
    Baseline + Local Branch Classifiers (gated residual, gate=0 at init)
    Optimizer: SAM (applied externally in training loop)
    """
    def __init__(self,num_classes=Config.NUM_CLASSES,embed_dim=Config.EMBED_DIM):
        super().__init__()
        self.backbone=ConvNeXtTiny(dpr=Config.DROP_PATH_RATE); self.backbone=load_convnext_pretrained(self.backbone)
        self.dims=[96,192,384,768]
        # Baseline components
        self.aux_heads=nn.ModuleList([ClassificationHead(d,num_classes) for d in self.dims])
        self.bottleneck=nn.Sequential(nn.Linear(768,embed_dim),nn.BatchNorm1d(embed_dim),nn.ReLU(True))
        self.classifier=nn.Linear(embed_dim,num_classes)
        # Novel: local branches
        self.local_branches=LocalBranchClassifiers(768,num_classes,Config.NUM_LOCAL)
        self.local_gate=nn.Parameter(torch.zeros(1))  # starts at 0 → baseline

    def forward(self,x,return_all=False):
        f,sts=self.backbone(x); slg=[h(s) for h,s in zip(self.aux_heads,sts)]
        base_emb=self.bottleneck(f)  # baseline embedding
        local_feat, branch_logits = self.local_branches(sts[-1])
        embedding=base_emb+self.local_gate*local_feat  # gated addition
        en=F.normalize(embedding,p=2,dim=1); lg=self.classifier(embedding)
        if return_all: return {'embedding':embedding,'embedding_normed':en,'logits':lg,'stage_logits':slg,
                               'branch_logits':branch_logits,'final_feature':f,'local_gate':self.local_gate.item()}
        return en, lg


# === SAM OPTIMIZER ===
class SAM(torch.optim.Optimizer):
    def __init__(self,params,base_optimizer_cls=torch.optim.SGD,rho=0.05,**kwargs):
        super().__init__(params,dict(rho=rho,**kwargs))
        self.base_optimizer=base_optimizer_cls(self.param_groups,**kwargs)
        self.param_groups=self.base_optimizer.param_groups; self.defaults.update(self.base_optimizer.defaults)
    @torch.no_grad()
    def first_step(self,zero_grad=False):
        gn=self._grad_norm()
        for g in self.param_groups:
            sc=g["rho"]/(gn+1e-12)
            for p in g["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"]=p.data.clone(); p.add_(p.grad*sc.to(p))
        if zero_grad: self.zero_grad()
    @torch.no_grad()
    def second_step(self,zero_grad=False):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                p.data=self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
    def _grad_norm(self):
        dev=self.param_groups[0]["params"][0].device
        return torch.stack([p.grad.norm(2).to(dev) for g in self.param_groups for p in g["params"] if p.grad is not None]).norm(2)
    def step(self,closure=None): raise NotImplementedError("Use first_step/second_step")
    def load_state_dict(self,sd): super().load_state_dict(sd); self.base_optimizer.param_groups=self.param_groups

class ModelEMA:
    def __init__(self,model,decay=0.9999):
        self.shadow={k:v.clone().float() for k,v in model.state_dict().items() if v.is_floating_point()}
        self.decay=decay; self.backup={}
    def update(self,model):
        with torch.no_grad():
            for k,v in model.state_dict().items():
                if k in self.shadow: self.shadow[k]=self.decay*self.shadow[k]+(1-self.decay)*v.float()
    def apply_shadow(self,model):
        self.backup={k:v.clone() for k,v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict({**model.state_dict(),**{k:v.to(next(model.parameters()).device) for k,v in self.shadow.items()}},strict=False)
    def restore(self,model): model.load_state_dict({**model.state_dict(),**self.backup},strict=False); self.backup={}

def centralize_gradients(opt):
    for g in opt.param_groups:
        for p in g['params']:
            if p.grad is None: continue
            gr=p.grad.data
            if len(gr.shape)>1: gr.add_(-gr.mean(dim=tuple(range(1,len(gr.shape))),keepdim=True))

# === TEACHER ===
class DINOv2Teacher(nn.Module):
    def __init__(self,ntb=2):
        super().__init__(); print("Loading DINOv2-base teacher...")
        self.model=torch.hub.load('facebookresearch/dinov2','dinov2_vitb14')
        for p in self.model.parameters(): p.requires_grad=False
        for blk in self.model.blocks[-ntb:]:
            for p in blk.parameters(): p.requires_grad=True
    @torch.no_grad()
    def forward(self,x):
        x=self.model.prepare_tokens_with_masks(x)
        for blk in self.model.blocks: x=blk(x)
        return self.model.norm(x)[:,0]

# === LOSSES ===
class TripletLoss(nn.Module):
    def __init__(self,m=0.3): super().__init__(); self.m=m
    def forward(self,e,labels):
        d=torch.cdist(e,e); l=labels.view(-1,1); pm=l.eq(l.T).float(); nm=l.ne(l.T).float()
        return F.relu((d*pm).max(1)[0]-(d*nm+pm*1e9).min(1)[0]+self.m).mean()

class SymmetricInfoNCELoss(nn.Module):
    def __init__(self,t=0.07): super().__init__(); self.t=t
    def forward(self,d,s,labels):
        d=F.normalize(d,1); s=F.normalize(s,1); sim=d@s.T/self.t; l=labels.view(-1,1); pm=l.eq(l.T).float()
        # Baseline formula: -log(sum(softmax*pos)/num_pos)
        l1=-torch.log((F.softmax(sim,1)*pm).sum(1)/pm.sum(1).clamp(1)).mean()
        l2=-torch.log((F.softmax(sim.T,1)*pm).sum(1)/pm.sum(1).clamp(1)).mean()
        return 0.5*(l1+l2)

class SelfDistillationLoss(nn.Module):
    def __init__(self,t=4.,w=[0.1,0.2,0.3,0.4]): super().__init__(); self.t=t; self.w=w
    def forward(self,sls):
        out=0.; f=sls[-1]
        for i in range(len(sls)-1): out+=self.w[i]*(self.t**2)*F.kl_div(F.log_softmax(f/self.t,1),F.softmax(sls[i]/self.t,1),reduction='batchmean')
        return out

class UAPALoss(nn.Module):
    def __init__(self,t=4.): super().__init__(); self.T0=t
    def forward(self,dl,sl):
        Ud=-(F.softmax(dl,1)*F.log_softmax(dl,1)).sum(1).mean(); Us=-(F.softmax(sl,1)*F.log_softmax(sl,1)).sum(1).mean()
        T=self.T0*(1+torch.sigmoid(Ud-Us)); return (T**2)*F.kl_div(F.log_softmax(dl/T,1),F.softmax(sl/T,1),reduction='batchmean')

class CrossDistillationLoss(nn.Module):
    def forward(self,sf,tf): sf=F.normalize(sf,1); tf=F.normalize(tf,1); return F.mse_loss(sf,tf)+(1-F.cosine_similarity(sf,tf).mean())

class GeoAllLoss(nn.Module):
    def __init__(self,cfg=Config):
        super().__init__(); self.cfg=cfg
        self.ce=nn.CrossEntropyLoss(label_smoothing=0.1); self.triplet=TripletLoss(cfg.MARGIN)
        self.nce=SymmetricInfoNCELoss(); self.sd=SelfDistillationLoss(cfg.TEMPERATURE)
        self.uapa=UAPALoss(cfg.BASE_TEMPERATURE); self.cdist=CrossDistillationLoss()
    def forward(self,do,so,labels,td=None,ts=None,pid_w=None):
        if pid_w is None: pid_w=defaultdict(lambda:1.0)
        L: Dict[str,Any]={}
        ce=0.
        for lg in do['stage_logits']: ce+=0.25*self.ce(lg,labels)
        ce+=self.ce(do['logits'],labels)
        for lg in so['stage_logits']: ce+=0.25*self.ce(lg,labels)
        ce+=self.ce(so['logits'],labels); L['ce']=pid_w['ce']*ce
        # Branch CE (very light weight)
        bce=0.
        for bl in do['branch_logits']: bce+=self.ce(bl,labels)
        for bl in so['branch_logits']: bce+=self.ce(bl,labels)
        L['branch_ce']=0.05*bce  # fixed 0.05 — does not dominate
        L['triplet']=self.cfg.LAMBDA_TRIPLET*pid_w['triplet']*(self.triplet(do['embedding_normed'],labels)+self.triplet(so['embedding_normed'],labels))
        L['csc']=self.cfg.LAMBDA_CSC*pid_w['csc']*self.nce(do['embedding_normed'],so['embedding_normed'],labels)
        L['self_dist']=self.cfg.LAMBDA_SELF_DIST*(self.sd(do['stage_logits'])+self.sd(so['stage_logits']))
        L['uapa']=self.cfg.LAMBDA_ALIGN*self.uapa(do['logits'],so['logits'])
        if td is not None: L['cross_dist']=self.cfg.LAMBDA_CROSS_DIST*(self.cdist(do['final_feature'],td)+self.cdist(so['final_feature'],ts))
        total=sum(L.values()); L['total']=total; return total,L

# === EVAL ===
def evaluate(model,test_ds,device,ema=None):
    if ema: ema.apply_shadow(model)
    model.eval()
    loader=DataLoader(test_ds,Config.BATCH_SIZE,False,num_workers=Config.NUM_WORKERS,pin_memory=True)
    df,dl=[],[]
    with torch.no_grad():
        for b in loader: f,_=model(b['drone'].to(device)); df.append(f.cpu()); dl.append(b['label'])
    df=torch.cat(df); dl=torch.cat(dl)
    tr=get_transforms("test"); sat_dir=os.path.join(test_ds.root,Config.SATELLITE_DIR); sf,sl=[],[]
    for loc in [f"{l:04d}" for l in Config.TRAIN_LOCS+Config.TEST_LOCS]:
        sp=os.path.join(sat_dir,loc,"0.png")
        if not os.path.exists(sp): continue
        t=tr(Image.open(sp).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad(): f,_=model(t)
        sf.append(f.cpu()); sl.append(test_ds.location_to_idx.get(loc,-1-len(sl)))
    sf=torch.cat(sf); sl=torch.tensor(sl)
    if ema: ema.restore(model)
    _,idx=(df@sf.T).sort(1,descending=True); N=len(df); r1=r5=r10=0; ap=0.
    for i in range(N):
        c=torch.where(sl[idx[i]]==dl[i])[0]
        if len(c)==0: continue
        fc=c[0].item()
        if fc<1: r1+=1
        if fc<5: r5+=1
        if fc<10: r10+=1
        ap+=sum((j+1)/(p.item()+1) for j,p in enumerate(c))/len(c)
    return {'R@1':r1/N*100,'R@5':r5/N*100,'R@10':r10/N*100},ap/N*100

class WarmupCosineScheduler:
    def __init__(self,opt,we,te,min_lr=1e-6):
        self.opt=opt; self.we=we; self.te=te; self.min_lr=min_lr; self.base_lr=opt.param_groups[0]['lr']
    def step(self,epoch):
        if epoch<self.we: lr=self.base_lr*(epoch+1)/self.we
        else:
            pr=(epoch-self.we)/(self.te-self.we); lr=self.min_lr+0.5*(self.base_lr-self.min_lr)*(1+math.cos(math.pi*pr))
        for pg in self.opt.param_groups: pg['lr']=lr; return lr

def train_one_epoch(model,teacher,loader,criterion,optimizer,ema,pid,device,epoch):
    model.train()
    if teacher: teacher.eval()
    total=0.; ls=defaultdict(float)
    for bi,batch in enumerate(loader):
        d=batch['drone'].to(device); s=batch['satellite'].to(device); labels=batch['label'].to(device)
        td,ts=None,None
        if teacher:
            with torch.no_grad(): td=teacher(d); ts=teacher(s)
        pid_w=pid.get_weights() if epoch>=Config.WARMUP_EPOCHS else None
        def compute_loss():
            with autocast("cuda",dtype=torch.bfloat16):
                do=model(d,return_all=True); so=model(s,return_all=True)
                return criterion(do,so,labels,td,ts,pid_w), do
        (loss1,L),do=compute_loss(); loss1.backward(); centralize_gradients(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.first_step(zero_grad=True)
        (loss2,_),_=compute_loss(); loss2.backward(); centralize_gradients(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.second_step(zero_grad=True)
        if ema: ema.update(model)
        if epoch>=Config.WARMUP_EPOCHS: pid.update({k:v.item() if torch.is_tensor(v) else v for k,v in L.items()})
        total+=loss1.item()
        for k,v in L.items(): ls[k]+=v.item() if torch.is_tensor(v) else v
        if bi%10==0: print(f"  B{bi}/{len(loader)} L={loss1.item():.4f} gate={do['local_gate']:.4f}")
    n=max(1,len(loader)); return total/n, {k:v/n for k,v in ls.items()}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=Config.NUM_EPOCHS)
    parser.add_argument("--batch_size",type=int,default=Config.BATCH_SIZE)
    parser.add_argument("--data_root",type=str,default=Config.DATA_ROOT)
    parser.add_argument("--test",action="store_true")
    args,_=parser.parse_known_args()
    Config.NUM_EPOCHS=args.epochs; Config.BATCH_SIZE=args.batch_size; Config.DATA_ROOT=args.data_root
    if args.test: Config.NUM_EPOCHS=1; Config.NUM_WORKERS=0; Config.BATCH_SIZE=8; Config.P=2
    Config.K=max(2,Config.BATCH_SIZE//Config.P)
    print("="*60); print("GeoAll: Unified SAM+LocalBranch+PID — SUES-200"); print("="*60)
    set_seed(Config.SEED); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(Config.OUTPUT_DIR,exist_ok=True)
    train_ds=SUES200Dataset(Config.DATA_ROOT,"train",transform=get_transforms("train"))
    test_ds=SUES200Dataset(Config.DATA_ROOT,"test",transform=get_transforms("test"))
    loader=DataLoader(train_ds,batch_sampler=PKSampler(train_ds,Config.P,Config.K),num_workers=Config.NUM_WORKERS,pin_memory=True)
    model=GeoAllStudent(num_classes=len(Config.TRAIN_LOCS)).to(device)
    print(f"  Student: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    try: teacher=DINOv2Teacher().to(device); teacher.eval()
    except: teacher=None; print("No teacher")
    criterion=GeoAllLoss()
    loss_names=['ce','branch_ce','triplet','csc','self_dist','uapa','cross_dist']
    pid=FuzzyPIDController(loss_names)
    optimizer=SAM(model.parameters(),base_optimizer_cls=torch.optim.SGD,rho=Config.SAM_RHO,lr=Config.LR,momentum=0.9,weight_decay=5e-4)
    scheduler=WarmupCosineScheduler(optimizer,Config.WARMUP_EPOCHS,Config.NUM_EPOCHS)
    ema=ModelEMA(model,Config.EMA_DECAY)
    best_r1=0.
    for epoch in range(Config.NUM_EPOCHS):
        lr=scheduler.step(epoch)
        print(f"\n{'='*40}\nEp {epoch+1}/{Config.NUM_EPOCHS} LR={lr:.6f}\n{'='*40}")
        avg_l,ld=train_one_epoch(model,teacher,loader,criterion,optimizer,ema,pid,device,epoch)
        print(f"  AvgL={avg_l:.4f}")
        for k,v in sorted(ld.items()):
            if k!='total': print(f"  {k}: {v:.4f}")
        if epoch>=Config.WARMUP_EPOCHS:
            w=pid.get_weights(); print(f"  PID: ce={w.get('ce',1.):.2f} trip={w.get('triplet',1.):.2f} csc={w.get('csc',1.):.2f}")
        if (epoch+1)%5==0 or epoch==Config.NUM_EPOCHS-1:
            rec,ap=evaluate(model,test_ds,device,ema)
            print(f"  R@1:{rec['R@1']:.2f}% R@5:{rec['R@5']:.2f}% R@10:{rec['R@10']:.2f}% AP:{ap:.2f}%")
            if rec['R@1']>best_r1:
                best_r1=rec['R@1']
                torch.save({'epoch':epoch,'model':model.state_dict(),'r1':best_r1},os.path.join(Config.OUTPUT_DIR,'geoall_best.pth'))
                print(f"  *** Best R@1={best_r1:.2f}% ***")
    print(f"\nDone! Best R@1={best_r1:.2f}%")

if __name__=="__main__": main()
