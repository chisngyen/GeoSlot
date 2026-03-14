# Ablation Study — SPDGeo-DPEA-MAR (EXP35) ✅ COMPLETE

**Full model**: EXP35 — 12 losses, 120 locs, 60 epochs per ablation

## Component Impact Ranking (Final)

| Rank | Component | Best R@1 w/o | Notes |
|------|-----------|-------------|-------|
| 1 | **ProxyAnchor** | 88.81% | Most critical — plateau at Ep20 |
| 2 | **MaskRecon** | 92.89% | MAE-style recon very impactful |
| 3 | **FusionGate** | 93.27% | Adaptive weighting matters |
| 4 | **AltPred** | 93.47% | Altitude prediction helps |
| 5 | **UAPA** | 94.43% | Uncertainty-aware distill |
| 6 | **EMA** | 94.46% | Moderate contribution |
| 7 | **AltConsistLoss** | 94.55% | Small but positive |
| 8 | **PartConsistency** | 94.62% | KL alignment minor |
| 9 | **Diversity** | 94.94% | Minimal impact |
| 10 | **SelfDistill** | 95.04% | Nearly no effect |
| 11 | **DeepAltFiLM** | 95.03% | Nearly no effect |
| 12 | **CrossDistill** | 95.28% | Slightly harmful? |

## All Results

| # | Group | Ablation | Best R@1 | Ep10 | Ep20 | Ep30 | Ep40 | Ep50 | Ep60 |
|---|-------|----------|----------|------|------|------|------|------|------|
| 1 | A | w/o ProxyAnchor | **88.81%** | 81.4 | 88.8 | 88.7 | 88.7 | 88.7 | 88.7 |
| 2 | A | w/o FusionGate | **93.27%** | 80.3 | 89.6 | 93.2 | 92.9 | 93.3 | 93.3 |
| 3 | B | w/o EMA | **94.46%** | 78.6 | 89.0 | 92.6 | 93.7 | 94.3 | 94.5 |
| 4 | B | w/o DeepAltFiLM | **95.03%** | 78.9 | 89.4 | 93.2 | 94.4 | 94.8 | 95.0 |
| 5 | C | w/o AltConsistLoss | **94.55%** | 79.0 | 89.2 | 92.7 | 94.1 | 94.3 | 94.5 |
| 6 | C | w/o MaskRecon | **92.89%** | 77.0 | 88.1 | 91.2 | 92.9 | 92.5 | 92.9 |
| 7 | D | w/o AltPred | **93.47%** | 76.4 | 86.4 | 91.8 | 93.5 | 93.3 | 93.4 |
| 8 | D | w/o Diversity | **94.94%** | 78.6 | 89.3 | 93.4 | 94.5 | 94.8 | 94.9 |
| 9 | E | w/o PartConsistency | **94.62%** | 78.7 | 89.1 | 93.2 | 94.2 | 94.4 | 94.6 |
| 10 | E | w/o CrossDistill | **95.28%** | 79.0 | 89.7 | 93.7 | 94.7 | 95.2 | 95.3 |
| 11 | F | w/o SelfDistill | **95.04%** | 79.1 | 89.2 | 93.7 | 94.4 | 94.9 | 95.0 |
| 12 | F | w/o UAPA | **94.43%** | 79.0 | 89.3 | 92.5 | 94.0 | 94.3 | 94.4 |

## Key Takeaways

**Essential (>1% drop):**
- ProxyAnchor, MaskRecon, FusionGate, AltPred

**Helpful (<1% drop):**
- UAPA, EMA, AltConsistLoss, PartConsistency

**Negligible / Possibly harmful:**
- Diversity, SelfDistill, DeepAltFiLM, CrossDistill (bỏ CrossDistill lại *tốt hơn* 95.28%)
