"""
MobileGeo: Cross-View Drone Geo-Localization
Single-file implementation for Kaggle with H100 80GB GPU
Based on paper: "MobileGeo: Exploring Hierarchical Knowledge Distillation for 
                Resource-Efficient Cross-view Drone Geo-Localization"
"""
import subprocess
import importlib

def pip_install(pkg, extra=""):
    subprocess.run(f"pip install -q {extra} {pkg}",
                   shell=True, capture_output=True, text=True)

print("[1/2] Installing packages...")
for p in ["timm", "tqdm"]:
    try:
        importlib.import_module(p)
    except ImportError:
        pip_install(p)
print("[2/2] Setup complete!")

# ============================================================================
# IMPORTS
# ============================================================================
import os
import math
import random
import argparse
import numpy as np
from typing import Dict, Any
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
import timm

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Paths - Update these for Kaggle
    DATA_ROOT = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
    DRONE_DIR = "drone-view"
    SATELLITE_DIR = "satellite-view"
    OUTPUT_DIR = "/kaggle/working"
    
    # Training
    NUM_WORKERS = 8
    P = 8  # Number of locations per batch
    K = 4  # Number of samples per location
    BATCH_SIZE = 256
    NUM_EPOCHS = 120
    LR = 0.001
    WARMUP_EPOCHS = 5
    
    # Model
    IMG_SIZE = 224
    NUM_CLASSES = 120  # 120 training locations (standard SUES-200 protocol)
    EMBED_DIM = 768
    DROP_PATH_RATE = 0.1
    
    # Distillation
    TEMPERATURE = 4.0
    BASE_TEMPERATURE = 4.0
    
    # Loss weights
    LAMBDA_TRIPLET = 1.0
    LAMBDA_CSC = 0.5
    LAMBDA_SELF_DIST = 0.5
    LAMBDA_CROSS_DIST = 0.3
    LAMBDA_ALIGN = 0.2
    MARGIN = 0.3
    
    # Altitude to use (150, 200, 250, 300)
    ALTITUDES = ["150", "200", "250", "300"]
    
    # Standard SUES-200 benchmark split: 120 train / 80 test (fixed)
    TRAIN_LOCS = list(range(1, 121))   # Locations 1-120 for training
    TEST_LOCS  = list(range(121, 201)) # Locations 121-200 for testing
    
    # Mixed precision
    USE_AMP = True
    
    SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    

# ============================================================================
# DATASET
# ============================================================================
class SUES200Dataset(Dataset):
    """SUES-200 Dataset for Cross-View Geo-Localization
    
    Standard SUES-200 benchmark protocol:
      - Training: locations 1-120 (120 locations)
      - Testing:  locations 121-200 (80 locations)
      - Fixed split (no random shuffle)
    """
    
    def __init__(self, root, mode="train", altitudes=None, transform=None,
                 train_locs=None, test_locs=None):
        self.root = root
        self.mode = mode
        self.altitudes = altitudes or Config.ALTITUDES
        self.transform = transform
        
        self.drone_dir = os.path.join(root, Config.DRONE_DIR)
        self.satellite_dir = os.path.join(root, Config.SATELLITE_DIR)
        
        # Use standard SUES-200 benchmark split (fixed, not random)
        if train_locs is None:
            train_locs = Config.TRAIN_LOCS
        if test_locs is None:
            test_locs = Config.TEST_LOCS
        
        # Convert location IDs to folder names (zero-padded 4 digits)
        loc_ids = train_locs if mode == "train" else test_locs
        self.locations = [f"{loc:04d}" for loc in loc_ids]
        self.location_to_idx = {loc: idx for idx, loc in enumerate(self.locations)}
        
        # Build samples list: (drone_path, satellite_path, location_idx, altitude)
        self.samples = []
        self.drone_by_location = defaultdict(list)
        
        for loc in self.locations:
            loc_idx = self.location_to_idx[loc]
            sat_path = os.path.join(self.satellite_dir, loc, "0.png")
            
            if not os.path.exists(sat_path):
                continue
                
            for alt in self.altitudes:
                alt_dir = os.path.join(self.drone_dir, loc, alt)
                if not os.path.isdir(alt_dir):
                    continue
                    
                for img_name in sorted(os.listdir(alt_dir)):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        drone_path = os.path.join(alt_dir, img_name)
                        self.samples.append((drone_path, sat_path, loc_idx, alt))
                        self.drone_by_location[loc_idx].append(len(self.samples) - 1)
        
        print(f"[{mode}] Loaded {len(self.samples)} samples from {len(self.locations)} locations")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        drone_path, sat_path, loc_idx, altitude = self.samples[idx]
        
        drone_img = Image.open(drone_path).convert('RGB')
        sat_img = Image.open(sat_path).convert('RGB')
        
        if self.transform:
            drone_img = self.transform(drone_img)
            sat_img = self.transform(sat_img)
            
        return {
            'drone': drone_img,
            'satellite': sat_img,
            'label': loc_idx,
            'altitude': int(altitude)
        }


class PKSampler:
    """P-K Sampler: P locations, K samples per location per batch"""
    
    def __init__(self, dataset, p=8, k=4):
        self.dataset = dataset
        self.p = p
        self.k = k
        self.locations = list(dataset.drone_by_location.keys())
        
    def __iter__(self):
        # Shuffle locations
        locations = self.locations.copy()
        random.shuffle(locations)
        
        batch = []
        for loc in locations:
            indices = self.dataset.drone_by_location[loc]
            if len(indices) < self.k:
                # Repeat if not enough samples
                indices = indices * (self.k // len(indices) + 1)
            sampled = random.sample(indices, self.k)
            batch.extend(sampled)
            
            if len(batch) >= self.p * self.k:
                yield batch[:self.p * self.k]
                batch = batch[self.p * self.k:]
                
    def __len__(self):
        return len(self.locations) // self.p


def get_transforms(mode="train"):
    if mode == "train":
        return T.Compose([
            T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomResizedCrop(Config.IMG_SIZE, scale=(0.8, 1.0)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# ============================================================================
# CONVNEXT-TINY BACKBONE
# ============================================================================
class LayerNorm(nn.Module):
    """LayerNorm supporting channels_first and channels_last"""
    
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:  # channels_first
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block"""
    
    def __init__(self, dim, drop_path_rate=0., layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny Backbone with multi-stage outputs"""
    
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init=1e-6):
        super().__init__()
        
        self.num_stages = 4
        self.dims = dims
        
        # Stem
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        # Downsampling layers
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample)
        
        # Stages
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXtBlock(dims[i], dp_rates[cur + j], layer_scale_init)
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]
        
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        """Forward with multi-stage outputs"""
        stage_outputs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            stage_outputs.append(x)
        
        # Global average pooling on final stage
        final_feat = x.mean([-2, -1])  # [B, C]
        final_feat = self.norm(final_feat)
        
        return final_feat, stage_outputs
    
    def forward(self, x):
        return self.forward_features(x)


def load_convnext_pretrained(model):
    """Load ImageNet-22K pretrained weights"""
    url = "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth"
    try:
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=True)
        state_dict = checkpoint["model"]
        # Filter out head weights
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
        model.load_state_dict(state_dict, strict=False)
        print("Loaded ConvNeXt-Tiny pretrained weights from ImageNet-22K")
    except Exception as e:
        print(f"Could not load pretrained weights: {e}")
    return model


# ============================================================================
# MOBILEGEO STUDENT MODEL
# ============================================================================
class ClassificationHead(nn.Module):
    """Auxiliary classification head for each stage"""
    
    def __init__(self, in_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.fc(x)


class GeneralizedMeanPooling(nn.Module):
    """Generalized Mean Pooling (GeM)"""
    
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        
    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)


class MobileGeoStudent(nn.Module):
    """MobileGeo Student Model with Self-Distillation Heads"""
    
    def __init__(self, num_classes, embed_dim=768, drop_path_rate=0.1):
        super().__init__()
        
        self.backbone = ConvNeXtTiny(drop_path_rate=drop_path_rate)
        self.backbone = load_convnext_pretrained(self.backbone)
        
        self.dims = [96, 192, 384, 768]
        self.embed_dim = embed_dim
        
        # Auxiliary classification heads for self-distillation (stages 1-4)
        self.aux_heads = nn.ModuleList([
            ClassificationHead(dim, num_classes) for dim in self.dims
        ])
        
        # Bottleneck for final embedding
        self.bottleneck = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # GeM pooling
        self.gem = GeneralizedMeanPooling()
        
    def forward(self, x, return_all=False):
        final_feat, stage_outputs = self.backbone(x)
        
        # Get logits from each stage for self-distillation
        stage_logits = [head(feat) for head, feat in zip(self.aux_heads, stage_outputs)]
        
        # Final embedding through bottleneck
        embedding = self.bottleneck(final_feat)
        embedding_normed = F.normalize(embedding, p=2, dim=1)
        
        # Classification logits
        logits = self.classifier(embedding)
        
        if return_all:
            return {
                'embedding': embedding,
                'embedding_normed': embedding_normed,
                'logits': logits,
                'stage_logits': stage_logits,
                'stage_features': stage_outputs,
                'final_feature': final_feat
            }
        
        return embedding_normed, logits


# ============================================================================
# DINOV2 TEACHER (for Cross-Distillation)
# ============================================================================
class DINOv2Teacher(nn.Module):
    """DINOv2-Base Teacher Model (frozen except last N blocks)"""
    
    def __init__(self, num_trainable_blocks=2):
        super().__init__()
        
        print("Loading DINOv2-base teacher model...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.num_channels = 768
        self.num_trainable_blocks = num_trainable_blocks
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze last N blocks
        for blk in self.model.blocks[-num_trainable_blocks:]:
            for param in blk.parameters():
                param.requires_grad = True
                
        print(f"DINOv2 loaded. Last {num_trainable_blocks} blocks are trainable.")
        
    @torch.no_grad()
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Prepare tokens
        x = self.model.prepare_tokens_with_masks(x)
        
        # Forward through all blocks
        for blk in self.model.blocks:
            x = blk(x)
            
        x = self.model.norm(x)
        
        # CLS token
        cls_token = x[:, 0]  # [B, 768]
        
        return cls_token


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class TripletLoss(nn.Module):
    """Triplet Loss with Hard Mining"""
    
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        # Compute pairwise distances
        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        
        N = embeddings.size(0)
        labels = labels.view(-1, 1)
        mask_pos = labels.eq(labels.T).float()
        mask_neg = labels.ne(labels.T).float()
        
        # Hard positive: max distance among positives
        dist_pos = dist_mat * mask_pos
        hard_pos = dist_pos.max(dim=1)[0]
        
        # Hard negative: min distance among negatives
        dist_neg = dist_mat * mask_neg + mask_pos * 1e9
        hard_neg = dist_neg.min(dim=1)[0]
        
        # Triplet loss
        loss = F.relu(hard_pos - hard_neg + self.margin)
        return loss.mean()


class SymmetricInfoNCELoss(nn.Module):
    """Cross-view Symmetric Contrastive Loss"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, drone_feats, sat_feats, labels):
        # Normalize features
        drone_feats = F.normalize(drone_feats, dim=1)
        sat_feats = F.normalize(sat_feats, dim=1)
        
        # Compute similarity
        sim_d2s = torch.mm(drone_feats, sat_feats.T) / self.temperature
        sim_s2d = sim_d2s.T
        
        B = drone_feats.size(0)
        
        # Create positive mask
        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T).float()
        
        # Drone -> Satellite loss
        loss_d2s = -torch.log(
            (F.softmax(sim_d2s, dim=1) * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)
        ).mean()
        
        # Satellite -> Drone loss
        loss_s2d = -torch.log(
            (F.softmax(sim_s2d, dim=1) * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)
        ).mean()
        
        return 0.5 * (loss_d2s + loss_s2d)


class SelfDistillationLoss(nn.Module):
    """Fine-Grained Inverse Self-Distillation Loss"""
    
    def __init__(self, temperature=4.0, weights=[0.1, 0.2, 0.3, 0.4]):
        super().__init__()
        self.temperature = temperature
        self.weights = weights
        
    def forward(self, stage_logits):
        """
        stage_logits: list of logits from each stage [z1, z2, z3, z4]
        Final stage z4 is the student, stages 1-3 are teachers
        """
        loss = 0.0
        final_logits = stage_logits[-1]  # z_N
        
        for i in range(len(stage_logits) - 1):
            teacher_logits = stage_logits[i]
            p_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
            p_student = F.log_softmax(final_logits / self.temperature, dim=1)
            
            kl_loss = F.kl_div(p_student, p_teacher, reduction='batchmean')
            loss += self.weights[i] * (self.temperature ** 2) * kl_loss
            
        return loss


class UAPALoss(nn.Module):
    """Uncertainty-Aware Prediction Alignment Loss"""
    
    def __init__(self, base_temperature=4.0):
        super().__init__()
        self.T0 = base_temperature
        
    def compute_uncertainty(self, logits):
        """Compute Shannon entropy as uncertainty measure"""
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        return entropy.mean()
    
    def forward(self, drone_logits, sat_logits):
        # Compute uncertainties
        U_drone = self.compute_uncertainty(drone_logits)
        U_sat = self.compute_uncertainty(sat_logits)
        
        # Adaptive temperature
        delta_U = U_drone - U_sat
        T = self.T0 * (1 + torch.sigmoid(delta_U))
        
        # KL divergence alignment (satellite teaches drone)
        p_sat = F.softmax(sat_logits / T, dim=1)
        p_drone = F.log_softmax(drone_logits / T, dim=1)
        
        loss = (T ** 2) * F.kl_div(p_drone, p_sat, reduction='batchmean')
        
        return loss


class CrossDistillationLoss(nn.Module):
    """Cross-Distillation from DINOv2 Teacher to Student"""
    
    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, student_feat, teacher_feat, student_logits=None, teacher_logits=None):
        # Feature distillation (MSE + Cosine)
        student_feat = F.normalize(student_feat, dim=1)
        teacher_feat = F.normalize(teacher_feat, dim=1)
        
        mse_loss = F.mse_loss(student_feat, teacher_feat)
        cosine_loss = 1 - F.cosine_similarity(student_feat, teacher_feat).mean()
        
        loss = mse_loss + cosine_loss
        
        return loss


class MobileGeoLoss(nn.Module):
    """Combined Loss for MobileGeo Training"""
    
    def __init__(self, num_classes, cfg=Config):
        super().__init__()
        self.cfg = cfg
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=cfg.MARGIN)
        self.csc_loss = SymmetricInfoNCELoss()
        self.self_dist_loss = SelfDistillationLoss(temperature=cfg.TEMPERATURE)
        self.uapa_loss = UAPALoss(base_temperature=cfg.BASE_TEMPERATURE)
        self.cross_dist_loss = CrossDistillationLoss(temperature=cfg.TEMPERATURE)
        
    def forward(self, drone_out, sat_out, labels, teacher_drone_feat=None, teacher_sat_feat=None):
        losses: Dict[str, Any] = {}
        
        # Classification loss (all stages + final)
        ce_loss = 0.0
        for i, logits in enumerate(drone_out['stage_logits']):
            ce_loss += 0.25 * self.ce_loss(logits, labels)
        ce_loss += self.ce_loss(drone_out['logits'], labels)
        
        for i, logits in enumerate(sat_out['stage_logits']):
            ce_loss += 0.25 * self.ce_loss(logits, labels)
        ce_loss += self.ce_loss(sat_out['logits'], labels)
        
        losses['ce'] = ce_loss
        
        # Triplet loss
        triplet_drone = self.triplet_loss(drone_out['embedding_normed'], labels)
        triplet_sat = self.triplet_loss(sat_out['embedding_normed'], labels)
        losses['triplet'] = self.cfg.LAMBDA_TRIPLET * (triplet_drone + triplet_sat)
        
        # Cross-view Symmetric Contrastive loss
        csc = self.csc_loss(drone_out['embedding_normed'], sat_out['embedding_normed'], labels)
        losses['csc'] = self.cfg.LAMBDA_CSC * csc
        
        # Self-distillation loss
        self_dist_drone = self.self_dist_loss(drone_out['stage_logits'])
        self_dist_sat = self.self_dist_loss(sat_out['stage_logits'])
        losses['self_dist'] = self.cfg.LAMBDA_SELF_DIST * (self_dist_drone + self_dist_sat)
        
        # UAPA loss
        uapa = self.uapa_loss(drone_out['logits'], sat_out['logits'])
        losses['uapa'] = self.cfg.LAMBDA_ALIGN * uapa
        
        # Cross-distillation loss (if teacher available)
        if teacher_drone_feat is not None:
            cross_dist_drone = self.cross_dist_loss(
                drone_out['final_feature'], teacher_drone_feat
            )
            cross_dist_sat = self.cross_dist_loss(
                sat_out['final_feature'], teacher_sat_feat
            )
            losses['cross_dist'] = self.cfg.LAMBDA_CROSS_DIST * (cross_dist_drone + cross_dist_sat)
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return total_loss, losses


# ============================================================================
# MSRM: MULTI-VIEW SELECTION REFINEMENT MODULE
# ============================================================================
class MSRM:
    """Multi-view Selection Refinement Module for inference"""
    
    def __init__(self, k=10, lambda_balance=0.5, temperature=1.0):
        self.k = k  # Number of views to select
        self.lambda_balance = lambda_balance
        self.temperature = temperature
        
    def compute_view_importance(self, features):
        """Compute information score for each view"""
        # features: [N_views, D]
        
        # Marginal entropy approximation (log variance)
        std_per_dim = features.std(dim=0)
        H_marginal = 0.5 * torch.log(2 * np.pi * np.e) + torch.log(std_per_dim.mean() + 1e-8)
        
        # Dynamic range entropy
        range_per_dim = features.max(dim=0)[0] - features.min(dim=0)[0]
        H_range = torch.log(range_per_dim.mean() + 1e-8)
        
        # Per-view importance
        importance_scores = []
        mean_feat = features.mean(dim=0)
        for i in range(features.size(0)):
            # Distance from mean (indicates how informative)
            dist_from_mean = torch.norm(features[i] - mean_feat)
            importance_scores.append(dist_from_mean)
            
        importance_scores = torch.stack(importance_scores)
        
        # Normalize
        importance_scores = (importance_scores - importance_scores.min()) / \
                           (importance_scores.max() - importance_scores.min() + 1e-8)
        
        return importance_scores
    
    def greedy_selection(self, features, importance_scores, positions=None):
        """Greedy selection balancing importance and spatial diversity"""
        N = features.size(0)
        if N <= self.k:
            return list(range(N))
        
        selected = []
        remaining = list(range(N))
        
        # Select first view (highest importance)
        first_idx = importance_scores.argmax().item()
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Greedy selection
        while len(selected) < self.k:
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining:
                # Importance term
                imp_score = importance_scores[idx]
                
                # Diversity term (distance to nearest selected)
                if positions is not None:
                    min_dist = min(
                        self._spatial_distance(positions[idx], positions[s])
                        for s in selected
                    )
                else:
                    min_dist = min(
                        torch.norm(features[idx] - features[s]).item()
                        for s in selected
                    )
                
                # Combined score
                score = self.lambda_balance * imp_score + (1 - self.lambda_balance) * min_dist
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            selected.append(best_idx)
            remaining.remove(best_idx)
            
        return selected
    
    def _spatial_distance(self, pos1, pos2):
        """Compute spatial distance between two positions (altitude, angle)"""
        h1, theta1 = pos1
        h2, theta2 = pos2
        
        # Altitude difference
        alt_diff = abs(h1 - h2) / 150  # Normalize by step size
        
        # Circular angle difference
        angle_diff = abs(theta1 - theta2)
        angle_diff = min(angle_diff, 360 - angle_diff) / 180  # Normalize
        
        return 2 * alt_diff + angle_diff
    
    def aggregate(self, features, selected_indices, importance_scores):
        """Weighted aggregation of selected views"""
        selected_feats = features[selected_indices]
        selected_importance = importance_scores[selected_indices]
        
        # Softmax weights
        weights = F.softmax(self.temperature * selected_importance, dim=0)
        
        # Weighted sum
        aggregated = (weights.unsqueeze(1) * selected_feats).sum(dim=0)
        
        return F.normalize(aggregated, dim=0)
    
    def __call__(self, multi_view_features, positions=None):
        """
        Args:
            multi_view_features: [N_views, D] features from multiple drone views
            positions: Optional list of (altitude, angle) tuples
        Returns:
            aggregated_feature: [D] refined feature
        """
        importance_scores = self.compute_view_importance(multi_view_features)
        selected_indices = self.greedy_selection(multi_view_features, importance_scores, positions)
        aggregated = self.aggregate(multi_view_features, selected_indices, importance_scores)
        
        return aggregated, selected_indices


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate(model, test_dataset, device, data_root=None):
    """Evaluate model on test set.
    
    Standard SUES-200 protocol:
      - Query: drone images from test locations (121-200)
      - Gallery: ALL 200 satellite images (confusion data from train + test)
      - This makes the retrieval harder and more realistic
    """
    model.eval()
    
    # Build test loader
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    # Extract drone query features (test locations only)
    all_drone_feats = []
    all_drone_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            drone_imgs = batch['drone'].to(device)
            labels = batch['label']
            
            drone_feats, _ = model(drone_imgs)
            
            all_drone_feats.append(drone_feats.cpu())
            all_drone_labels.append(labels)
    
    all_drone_feats = torch.cat(all_drone_feats, dim=0)
    all_drone_labels = torch.cat(all_drone_labels, dim=0)
    
    # Build FULL satellite gallery (ALL 200 locations = confusion data)
    # Per SUES-200 paper: gallery includes train locations as distractors
    transform = get_transforms("test")
    root = data_root or test_dataset.root
    satellite_dir = os.path.join(root, Config.SATELLITE_DIR)
    
    # Get ALL satellite locations (1-200)
    all_loc_ids = Config.TRAIN_LOCS + Config.TEST_LOCS
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
            # Use the test_dataset's label if this is a test location,
            # otherwise assign a unique negative label for train distractors
            if loc in test_dataset.location_to_idx:
                sat_labels_list.append(test_dataset.location_to_idx[loc])
            else:
                sat_labels_list.append(-1 - len(gallery_loc_names))  # unique negative
            gallery_loc_names.append(loc)
    
    sat_feats = torch.cat(sat_feats_list, dim=0)
    sat_labels = torch.tensor(sat_labels_list)
    
    print(f"  Gallery: {len(sat_feats)} satellite images (confusion data)")
    print(f"  Queries: {len(all_drone_feats)} drone images")
    
    # Drone -> Satellite retrieval
    recall_at_k, ap = compute_metrics(
        all_drone_feats, sat_feats, all_drone_labels, sat_labels
    )
    
    return recall_at_k, ap


def compute_metrics(query_feats, gallery_feats, query_labels, gallery_labels):
    """Compute Recall@K and mAP"""
    # Compute similarity matrix
    sim_matrix = torch.mm(query_feats, gallery_feats.T)
    
    # Sort by similarity (descending)
    _, indices = sim_matrix.sort(dim=1, descending=True)
    
    # Compute metrics
    N = query_feats.size(0)
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    ap_sum = 0
    
    for i in range(N):
        query_label = query_labels[i]
        ranked_labels = gallery_labels[indices[i]]
        
        # Find positions of correct matches
        correct_mask = ranked_labels == query_label
        correct_positions = torch.where(correct_mask)[0]
        
        if len(correct_positions) == 0:
            continue
            
        first_correct = correct_positions[0].item()
        
        # Recall@K
        if first_correct < 1:
            recall_at_1 += 1
        if first_correct < 5:
            recall_at_5 += 1
        if first_correct < 10:
            recall_at_10 += 1
            
        # AP
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


# ============================================================================
# TRAINING
# ============================================================================
class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr


def train_one_epoch(model, teacher, train_loader, criterion, optimizer, 
                    scaler, device, epoch, cfg=Config):
    model.train()
    if teacher is not None:
        teacher.eval()
    
    total_loss = 0
    loss_dict_sum = defaultdict(float)
    
    for batch_idx, batch in enumerate(train_loader):
        drone_imgs = batch['drone'].to(device)
        sat_imgs = batch['satellite'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=cfg.USE_AMP):
            # Student forward
            drone_out = model(drone_imgs, return_all=True)
            sat_out = model(sat_imgs, return_all=True)
            
            # Teacher forward (if available)
            teacher_drone_feat = None
            teacher_sat_feat = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_drone_feat = teacher(drone_imgs)
                    teacher_sat_feat = teacher(sat_imgs)
            
            # Compute loss
            loss, loss_dict = criterion(
                drone_out, sat_out, labels, 
                teacher_drone_feat, teacher_sat_feat
            )
        
        # Backward
        if cfg.USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_dict_sum[k] += v.item() if torch.is_tensor(v) else v
        
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    if len(train_loader) == 0:
        print("Warning: Empty dataloader!")
        return 0.0, {}
        
    avg_loss = total_loss / len(train_loader)
    avg_loss_dict = {k: v / len(train_loader) for k, v in loss_dict_sum.items()}
    
    return avg_loss, avg_loss_dict


def main():
    parser = argparse.ArgumentParser(description="Baseline MobileGeo Training")
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--data_root", type=str, default=Config.DATA_ROOT)
    parser.add_argument("--test", action="store_true", help="Run a quick smoke test")
    args, _ = parser.parse_known_args()

    Config.NUM_EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.DATA_ROOT = args.data_root
    
    if args.test:
        print(">>> SMOKE TEST MODE <<<")
        Config.NUM_EPOCHS = 1
        Config.NUM_WORKERS = 0
        Config.BATCH_SIZE = 8
        Config.P = 2
    
    Config.K = max(2, Config.BATCH_SIZE // Config.P)

    print("=" * 60)
    print("Baseline MobileGeo Training - SUES-200 Dataset")
    print("=" * 60)
    
    # Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    print("\nLoading dataset...")
    train_transform = get_transforms("train")
    test_transform = get_transforms("test")
    
    # Standard SUES-200 benchmark: fixed 120/80 split
    print(f"  Train locations: {len(Config.TRAIN_LOCS)} (IDs {Config.TRAIN_LOCS[0]}-{Config.TRAIN_LOCS[-1]})")
    print(f"  Test  locations: {len(Config.TEST_LOCS)} (IDs {Config.TEST_LOCS[0]}-{Config.TEST_LOCS[-1]})")
    
    train_dataset = SUES200Dataset(
        Config.DATA_ROOT, mode="train", 
        transform=train_transform,
    )
    test_dataset = SUES200Dataset(
        Config.DATA_ROOT, mode="test",
        transform=test_transform,
    )
    
    # Number of training classes = number of training locations
    num_classes = len(Config.TRAIN_LOCS)
    print(f"Number of training classes: {num_classes}")
    
    # DataLoaders
    train_sampler = PKSampler(train_dataset, p=Config.P, k=Config.K)
    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Models
    print("\nBuilding models...")
    model = MobileGeoStudent(num_classes=num_classes, embed_dim=Config.EMBED_DIM).to(device)
    
    # Try to load DINOv2 teacher
    try:
        teacher = DINOv2Teacher(num_trainable_blocks=2).to(device)
    except Exception as e:
        print(f"Could not load DINOv2 teacher: {e}")
        print("Training without cross-distillation")
        teacher = None
    
    # Loss and optimizer
    criterion = MobileGeoLoss(num_classes=num_classes)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=Config.LR,
        momentum=0.9,
        weight_decay=5e-4
    )
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=Config.WARMUP_EPOCHS,
        total_epochs=Config.NUM_EPOCHS
    )
    scaler = GradScaler(enabled=Config.USE_AMP)
    
    # Training loop
    print("\nStarting training...")
    best_recall = 0.0
    
    for epoch in range(Config.NUM_EPOCHS):
        lr = scheduler.step(epoch)
        print(f"\n{'='*40}")
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, LR: {lr:.6f}")
        print("=" * 40)
        
        avg_loss, loss_dict = train_one_epoch(
            model, teacher, train_loader, criterion, 
            optimizer, scaler, device, epoch
        )
        
        print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        for k, v in loss_dict.items():
            print(f"  {k}: {v:.4f}")
        
        # Evaluate
        if (epoch + 1) % 5 == 0 or epoch == Config.NUM_EPOCHS - 1:
            print("\nEvaluating...")
            recall_at_k, ap = evaluate(model, test_dataset, device)
            print(f"  R@1: {recall_at_k['R@1']:.2f}%")
            print(f"  R@5: {recall_at_k['R@5']:.2f}%")
            print(f"  R@10: {recall_at_k['R@10']:.2f}%")
            print(f"  mAP: {ap:.2f}%")
            
            # Save best model
            if recall_at_k['R@1'] > best_recall:
                best_recall = recall_at_k['R@1']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'recall@1': recall_at_k['R@1'],
                    'ap': ap
                }, os.path.join(Config.OUTPUT_DIR, 'best_model.pth'))
                print(f"  Saved best model!")
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best R@1: {best_recall:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()



