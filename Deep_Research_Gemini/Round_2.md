# Advancing Drone-to-Satellite Geo-Localization: Object-Centric Learning, VLM Distillation, and Extreme Efficiency on the SUES-200 Benchmark

## Introduction and Architectural Context

Cross-view geo-localization between Unmanned Aerial Vehicles (UAVs) and satellite platforms represents a highly complex domain within computer vision, characterized by extreme disparities in spatial geometry, illumination, scale, and perspective. The SUES-200 benchmark serves as the premier testing environment for these capabilities. Unlike legacy datasets, SUES-200 explicitly models altitude variations by providing drone imagery captured at 150m, 200m, 250m, and 300m, mapped against a unified satellite gallery.1 These inherent altitude shifts induce severe scale ambiguities and perspective distortions, rigorously testing an architecture's capacity to extract viewpoint-invariant representations.3

The current baseline architecture under consideration utilizes a ConvNeXt-Tiny backbone augmented with DINOv2 self-supervised distillation, achieving a respectable Recall@1 (R@1) of 82.35%. While ConvNeXt provides robust hierarchical convolutional feature extraction and DINOv2 offers excellent dense patch-level geometric understanding, the 82.35% metric indicates that the model struggles with the "confusion gallery"—the presence of hard negative geographical locations that possess nearly identical visual topologies (e.g., repeating suburban cul-de-sacs or identical university dormitories).5

An exhaustive analysis of the 2024–2025 academic literature reveals that state-of-the-art (SOTA) methodologies now routinely exceed 95% R@1 on the SUES-200 benchmark.5 Bridging the gap from 82.35% to the upper 90th percentile requires moving beyond standard metric learning. It necessitates the integration of object-centric reasoning via Slot Attention, the distillation of zero-shot spatial grounding from advanced Vision-Language Models (VLMs) like SigLIP2, the deployment of dynamic cross-view attention modules, and the utilization of extreme quantization for edge deployment.8 This report synthesizes these cutting-edge techniques into a comprehensive roadmap designed to maximize retrieval accuracy and inference efficiency.

## ---

1. Cross-View Geo-Localization State-of-the-Art on SUES-200 (2023-2025)

The trajectory of performance on the SUES-200 benchmark demonstrates a rapid evolution from rigid spatial parsing to highly dynamic, distillation-based, and error-controlled architectures. Understanding the current SOTA establishes the mathematical and architectural targets required to exceed 85% R@1.5

### 1.1. The Evolution of Spatial Partitioning

| Paper Metadata | Details                                                                                                           |
| -------------- | ----------------------------------------------------------------------------------------------------------------- |
| Title          | UAV-Satellite Cross-View Image Matching Based on Adaptive Threshold-Guided Ring Partitioning Framework            |
| Authors        | Y. Chen, et al.                                                                                                   |
| Venue & Year   | Remote Sensing (MDPI), 2025                                                                                       |
| Key Technique  | Adaptive Threshold-guided Ring Partitioning Framework (ATRPF) using heatmap-guided dynamic region adjustment.     |
| Metrics        | 73.72% R@1 and 76.93% AP on SUES-200 drone-to-satellite tasks.                                                    |
| Relevance      | Highlights the limitation of rigid spatial assumptions in the baseline and necessitates adaptive spatial pooling. |

Early paradigms, most notably the Local-aware Part Network (LPN), relied on a square-ring feature segmentation strategy. This method operated on the assumption that the primary object of interest resides strictly at the image center, with contextual environment radiating outward in concentric squares.12 On the SUES-200 dataset, LPN achieves highly constrained performance, yielding 61.58% R@1 at 150m and 81.47% at 300m.5 The rigid partitioning fragments continuous semantic structures—such as long buildings or roads—leading to erroneous feature extraction when drone altitudes fluctuate.12

To resolve this, the Adaptive Threshold-guided Ring Partitioning Framework (ATRPF) was introduced.13 ATRPF discards fixed geometrical rules, utilizing heatmap-guided adaptive thresholds and learnable hyperparameters to dynamically adjust the extraction regions based on the scene's actual semantic layout.13 While ATRPF improves upon LPN, yielding 73.72% R@1, it remains bounded by the limitations of pure convolutional partitioning without cross-attention mechanisms.14

### 1.2. High-Performance Architectures: AGEN and MCFA

| Paper Metadata | Details                                                                                                |
| -------------- | ------------------------------------------------------------------------------------------------------ |
| Title          | AGEN: An Adaptive Error Control and Global-Local Feature Fusion Network for UAV Geo-Localization       |
| Authors        | X. Liu, et al.                                                                                         |
| Venue & Year   | IEEE Sensors / PMC, 2025                                                                               |
| Key Technique  | DINOv2 integration coupled with a Fuzzy PID controller for Adaptive Error Control during pre-training. |
| Metrics        | 94.38% R@1 (150m), 95.75% R@1 (250m), 97.12% R@1 (300m) on SUES-200.                                   |
| Relevance      | Directly proves that augmenting a DINOv2 baseline with dynamic loss modulation yields >95% R@1.        |

The AGEN architecture represents a monumental leap in SUES-200 performance.5 AGEN pairs DINOv2 global feature extraction with multiple local classifier branches. The defining innovation of AGEN is its Adaptive Error Control (AEC) module, which utilizes a Fuzzy Proportional-Integral-Derivative (PID) controller to optimize the loss function dynamically.5 The proportional component reacts to immediate epoch errors, the integral component captures long-term training bias, and the derivative component evaluates the velocity of loss change to suppress oscillations.5 This mechanism makes the model highly resilient to the "confusion gallery" and extreme visual disturbances, allowing it to peak at 97.12% R@1 at the 300m altitude.5

| Paper Metadata | Details                                                                                                         |
| -------------- | --------------------------------------------------------------------------------------------------------------- |
| Title          | Multi-Scale Cascade and Feature Adaptive Alignment Network for Cross-View Geo-Localization                      |
| Authors        | Z. Wang, et al.                                                                                                 |
| Venue & Year   | Sensors (MDPI), 2025                                                                                            |
| Key Technique  | Multi-Scale Cascade Module (MSCM) and Feature Adaptive Alignment Module (FAAM) via cross-dimensional weighting. |
| Metrics        | +1.52% average R@1 improvement over previous SOTA on SUES-200.                                                  |
| Relevance      | Validates the use of ConvNeXt-Tiny as a backbone when paired with advanced multi-scale feature alignment.       |

Concurrently, the Multi-Scale Cascade and Feature Adaptive Alignment (MCFA) network utilizes the exact same ConvNeXt-Tiny backbone as the target baseline but achieves vastly superior results.15 MCFA deploys a Multi-Scale Cascade Module (MSCM) that actively captures features from adjacent regions and fuses them to understand intricate spatial relationships, entirely resolving the fragmentation issues of standard LPNs.12 Furthermore, its Feature Adaptive Alignment Module (FAAM) utilizes a cross-dimensional dynamic weighting strategy to adaptively reweight spatial and channel dependencies, successfully aligning the drone and satellite representations into a unified embedding space.15

### 1.3. The Pinnacle of Accuracy and Efficiency: MobileGeo and PFED

| Paper Metadata | Details                                                                                                                    |
| -------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Title          | MobileGeo: Exploring Hierarchical Knowledge Distillation for Resource-Efficient Cross-View Drone Geo-Localization          |
| Authors        | S. Hu, et al.                                                                                                              |
| Venue & Year   | arXiv, 2025                                                                                                                |
| Key Technique  | Hierarchical Distillation (HD-CVGL) and Uncertainty-Aware Prediction Alignment (UAPA).                                     |
| Metrics        | 97.87% R@1 at 300m on SUES-200; operates at 1022 FPS with 4.45G FLOPs.                                                     |
| Relevance      | Demonstrates the ultimate potential of the baseline if hierarchical distillation is applied to compress a massive teacher. |

For edge deployment, modern SOTA focuses on the intersection of maximal retrieval accuracy and minimal computational overhead. MobileGeo establishes a new benchmark by utilizing a Hierarchical Distillation (HD-CVGL) paradigm.16 Coupled with Uncertainty-Aware Prediction Alignment (UAPA), it distills massive amounts of spatial information into a compact student model.16 During inference, a Multi-view Selection Refinement Module (MSRM) uses mutual information to filter redundant views.16 MobileGeo achieves an astonishing 97.87% R@1 on the SUES-200 300m evaluation while requiring only 4.45G FLOPs—a remarkable 20$\times$ reduction in computational load compared to massive transformer ensembles, allowing it to process 1022 frames per second.16

Similarly, the PFED architecture reports dominant scores, matching the 97.87% R@1 threshold at high altitudes and establishing a definitive benchmark for both accuracy and edge-device viability.17

### 1.4. Summary of SOTA on SUES-200

To provide a clear mathematical landscape of the SUES-200 Drone ![]() Satellite retrieval task, the following metrics illustrate the progression of architectures:

| Model Architecture         | 150m R@1 | 200m R@1 | 250m R@1 | 300m R@1 | FLOPs / Efficiency Profile      |
| -------------------------- | -------- | -------- | -------- | -------- | ------------------------------- |
| Baseline (ConvNeXt+DINOv2) | ~82.35%  | ~82.35%  | ~82.35%  | ~82.35%  | Moderate (Standard FP32)        |
| LPN5                       | 61.58%   | 70.85%   | 80.38%   | 81.47%   | Low (CNN-based)                 |
| CCR5                       | 87.08%   | 93.57%   | 95.42%   | 96.82%   | High (Heavy correlation matrix) |
| AGEN5                      | 94.38%   | 91.78%   | 95.75%   | 97.12%   | High (DINOv2 + LPN branches)    |
| MobileGeo / PFED16         | >95%     | >95%     | >95%     | 97.87%   | Ultra-Low (4.45G FLOPs)         |

The data implies a definitive conclusion: the baseline architecture’s limitation is not the ConvNeXt backbone nor the DINOv2 weights, but rather the lack of adaptive spatial pooling, hierarchical distillation, and dynamic error control.

## ---

2. Slot Attention and Object-Centric Learning for Matching Tasks

Traditional cross-view matching relies heavily on global image descriptors (such as NetVLAD) or dense local patch comparisons. These approaches degrade rapidly under severe viewpoint shifts because a drone capturing a building facade processes entirely different texture and color distributions than a satellite capturing the flat roof of the exact same building. Object-centric learning, facilitated by Slot Attention, resolves this discrepancy by decomposing scenes into identifiable, permutation-invariant abstract entities known as "slots".18 This allows the architecture to match the semantic concept and geometry of a structure rather than its raw pixel statistics.

### 2.1. Probabilistic Slot Attention (NeurIPS 2024)

| Paper Metadata | Details                                                                                                                    |
| -------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Title          | Identifiable Object-Centric Representation Learning via Probabilistic Slot Attention                                       |
| Authors        | A. Kori, F. Locatello, et al.                                                                                              |
| Venue & Year   | NeurIPS, 2024                                                                                                              |
| Key Technique  | Imposes an aggregate mixture prior over slots using Gaussian Mixture Models (GMMs) for theoretical identifiability.        |
| Metrics        | Demonstrated superior Identifiable Slot Similarity (SIS) on standard imaging datasets (SPRITEWORLD, CLEVR).                |
| Relevance      | Solves the theoretical bottleneck of matching slots across disparate views by guaranteeing identical latent space mapping. |

Standard slot attention utilizes competitive cross-attention (induced by a softmax operation) to drive each slot to exclusively take ownership of one object.18 However, applying this to cross-view correspondence introduces a critical flaw: without supervision, there is no theoretical guarantee that Slot 1 in the drone view will represent the same building as Slot 1 in the satellite view.

Probabilistic Slot Attention (PSA) rectifies this identifiability problem.8 PSA augments standard slot attention by introducing per-datapoint Gaussian Mixture Models (GMMs) to cluster features into distinct object distributions.8 Agreement between the input features (keys) and the slots (queries) is measured by evaluating the normalized probability density of each key under the Gaussian model defined by each slot.8 The resulting attention matrix represents the true posterior probability of slot-to-feature assignment.8

By marginalizing data across the entire dataset, PSA induces a global aggregate posterior that provides theoretical identifiability guarantees up to an affine transformation and slot permutation.8 For the SUES-200 benchmark, this means that a slot representing a specific architectural structure in the satellite view is mathematically forced into the same latent subspace as the corresponding structure in the drone view, enabling mathematically rigorous instance-level matching.20 Furthermore, PSA utilizes Automatic Relevance Determination (ARD) to dynamically prune inactive slots, ensuring the model scales efficiently without hallucinating objects in sparse geographic areas.8

### 2.2. Disentangled Slot Attention (2025)

| Paper Metadata | Details                                                                                                                                        |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Title          | Learning global object-centric representations via disentangled slot attention                                                                 |
| Authors        | T. Chen, Y. Huang, et al.                                                                                                                      |
| Venue & Year   | Machine Learning (Springer), 2025                                                                                                              |
| Key Technique  | Disentangled Slot Attention (DISA) explicitly partitions latent slot dimensions into non-overlapping subsets for shape, texture, and position. |
| Metrics        | State-of-the-art scene decomposition and object identification across varying 2D/3D environments.                                              |
| Relevance      | Allows the baseline model to ignore textural mismatches (facade vs. roof) and compute similarity solely on geometric shape.                    |

The most persistent optical challenge in drone-to-satellite localization is the variance in observable material attributes. Disentangled Slot Attention (DISA) enhances object-centric learning by explicitly disentangling these attributes within the latent space.22 DISA partitions the semantic factors within the slot vectors, creating strictly non-overlapping subsets of latent dimensions dedicated independently to shape, texture, and extrinsic factors like position and scale.18

The application to the baseline architecture is highly strategic: by implementing DISA, the retrieval system can be programmed to compute cosine similarity exclusively on the latent subset dedicated to shape and positional relations, effectively blinding the network to viewpoint-dependent textural changes.23 The model will correctly retrieve a satellite image based on the geometric arrangement of building footprints, entirely disregarding the fact that the drone observes brick facades while the satellite observes gray asphalt roofs.24

### 2.3. Quality-Guided K-Adaptive Slot Attention (QASA, 2026)

| Paper Metadata | Details                                                                                                                 |
| -------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Title          | QASA: Quality-Guided K-Adaptive Slot Attention for Unsupervised Object-Centric Learning                                 |
| Authors        | T. Ouyang, et al.                                                                                                       |
| Venue & Year   | arXiv, 2026 (Preprint)                                                                                                  |
| Key Technique  | Unsupervised Slot-Quality metric decoupling slot selection from reconstruction.                                         |
| Metrics        | Outperforms existing K-adaptive baselines by 8.4% on average across real and synthetic datasets.                        |
| Relevance      | Prevents the model from wasting slots on broad, diffuse background features like large patches of grass or empty roads. |

While PSA handles dynamic slot counts probabilistically, QASA introduces a deterministic, quality-based metric to guide slot-object binding.26 QASA decouples slot selection from image reconstruction by evaluating an unsupervised "Slot-Quality metric"—defined strictly as the proportion of a slot's attention mass concentrated within its defined winning region.26 This mechanism prevents slots from binding to diffuse, broad background representations.26 For SUES-200, integrating QASA ensures that the limited slot budget focuses purely on fine-grained target features (e.g., distinct landmarks, road intersections) rather than interpolating massive, non-informative patches of vegetation or water.26

## ---

3. SigLIP2 and VLM Distillation Strategies

Large Vision-Language Models (VLMs) possess immense zero-shot capabilities and profound spatial reasoning, yet their massive parameter counts (often exceeding 10B) preclude them from real-time edge deployment on UAVs. Distilling the robust spatial grounding of modern VLMs into the lightweight ConvNeXt-Tiny baseline is a highly effective vector for performance enhancement.

### 3.1. SigLIP2 vs. CLIP and DINOv2: Zero-Shot Localization

| Paper Metadata | Details                                                                                                                              |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Title          | SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features               |
| Authors        | M. Tschannen, et al.                                                                                                                 |
| Venue & Year   | arXiv, February 2025                                                                                                                 |
| Key Technique  | Unification of LocCa (decoder-based captioning), SILC (self-distillation), and Masked Prediction into a single training recipe.      |
| Metrics        | Significant absolute percentage point gains over SigLIP/CLIP in dense prediction (segmentation +5 mIoU) and RefCOCO localization.    |
| Relevance      | Provides the ultimate teacher model for the baseline, offering both the semantic alignment of CLIP and the dense geometry of DINOv2. |

Released in early 2025, SigLIP2 significantly advances upon the architectures of CLIP, DINOv2, and the original SigLIP by unifying multiple, independently developed pretraining paradigms into a cohesive recipe.10

Standard CLIP models rely purely on a softmax contrastive loss. While exceptional at global semantic alignment, they exhibit a "bag-of-words" behavior that destroys fine-grained spatial cues.10 Conversely, DINOv2 utilizes self-supervised masked image modeling to generate rich, pixel-level dense features but completely lacks paired text supervision and semantic understanding.29

SigLIP2 bridges this spatial blindspot through LocCa (Localization and Captioning). During pre-training, a lightweight transformer decoder is attached to the un-pooled representations of the vision encoder.10 This decoder optimizes autoregressive captioning, referring expression localization (predicting exact bounding box coordinates for text descriptions), and grounded captioning.10 Once trained, the decoder heads are discarded, leaving a standalone vision encoder with profound native spatial awareness.10 Furthermore, SigLIP2 incorporates SILC (local-to-global consistency via an Exponential Moving Average teacher) and masked patch modeling to refine un-pooled feature representations.10 Finally, the NaFlex variant of SigLIP2 preserves the input's native aspect ratio and supports 2D Rotary Positional Embeddings (2D-RoPE), ensuring that geometric topography is never distorted by forced square resizing.10

### 3.2. VLM Distillation: AMoE and ARKD

Distilling SigLIP2 into a ConvNeXt-Tiny architecture requires sophisticated strategies beyond simple logit matching. The 2025 state-of-the-art leverages a Mixture-of-Teachers approach via the Asymmetric Mixture of Experts (AMoE) framework, combined with specific distillation granularities.30

1. Multi-Teacher Instantiation: The optimal strategy utilizes both DINOv3 (or DINOv2) and SigLIP2 as complementary teachers.30 DINO provides superior intra-image geometry-patch representations, while SigLIP2 provides text-aligned semantic grounding. The student model (ConvNeXt-Tiny) utilizes teacher-specific, single-layer MLP projection heads to translate its features into each teacher's embedding space.30
2. Patch-Level vs. Token-Level Distillation:

* Token-Level: The student's global summary token is trained to match the attention pooling layer output of the SigLIP2 teacher using a standard cosine similarity loss, transferring global semantic context.30
* Patch-Level: To preserve strict spatial grounding, the student's individual spatial tokens are explicitly mapped to the un-pooled dense features of the SigLIP2 teacher. This ensures the student learns fine-grained local representations.31 A token-balanced batching technique via FlexAttention ensures stable representation transfer across images of varying resolutions.30

3. Attention-Map Distillation via ARKD: Point-wise alignment is insufficient to capture the structural layout of a scene. The Asymmetric Relation Knowledge Distillation (ARKD) loss minimizes the divergence between the student's internal patch-to-patch attention maps and the teacher's patch-to-patch correlation matrices.30 By matching the pairwise geometry among samples, the lightweight ConvNeXt model is forced to group pixels into the identical semantic clusters utilized by SigLIP2, effectively transferring the VLM's object-centric spatial grounding without requiring its massive parameter count.30

## ---

4. Cross-View Attention Mechanisms

Even with perfectly distilled spatial features, the extreme domain gap between the oblique angles of a UAV and the orthographic, top-down views of a satellite requires explicit geometric alignment.

### 4.1. AttenGeo: CVCAM and MHSAM (ICLR 2025)

| Paper Metadata | Details                                                                                                                              |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Title          | Improving Cross-view Object Geo-localization: A Dual Attention Approach with Cross-view Interaction and Multi-Scale Spatial Features |
| Authors        | X. Ling, Y. Zhu                                                                                                                      |
| Venue & Year   | ICLR, 2025                                                                                                                           |
| Key Technique  | Cross-view and Cross-attention Module (CVCAM) coupled with a Multi-head Spatial Attention Module (MHSAM).                            |
| Metrics        | Achieves SOTA localization accuracy on CVOGL and the newly introduced G2D (Ground-to-Drone) datasets.                                |
| Relevance      | Provides a direct architectural drop-in to replace standard global pooling, allowing iterative cross-view feature alignment.         |

The AttenGeo architecture introduces two vital attention modules designed specifically to bridge drastic viewpoint discrepancies.3

Cross-view and Cross-attention Module (CVCAM): Traditional methods rely on simple feature concatenation or global average pooling, which fails to map complex structural transformations. CVCAM, instead, facilitates a deep, iterative information exchange. Features from the query image (drone) and reference image (satellite) are flattened and passed through a series of Cross-Attention Blocks (CABs) for ![]() iterations.9 This iterative cross-attention establishes implicit spatial correspondences between the drastic viewpoint changes, acting as a dynamic zero-shot correspondence estimator.3 Crucially, CVCAM aggressively suppresses irrelevant background edge noise by continually focusing the attention mass on the contextual information of the query object.3

Multi-head Spatial Attention Module (MHSAM): Operating sequentially after CVCAM, MHSAM refines the fused cross-view representations.9 It deploys three distinct attention heads containing convolutional and deconvolutional kernels of varying sizes to extract multi-scale spatial features.3 By processing the implicit correspondences through varied receptive fields, MHSAM isolates the precise local geometry of the query object irrespective of its orientation or scale in the aerial view.9 Integrating CVCAM and MHSAM into the ConvNeXt baseline will force the model to explicitly calculate the spatial transformations between the drone facade and the satellite footprint.

### 4.2. Geometric Alignment: Video2BEV Transformation

| Paper Metadata | Details                                                                                                                                |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Title          | Video2BEV: Transforming Drone Videos to BEVs for Video-based Geo-localization                                                          |
| Authors        | H. Ju, et al.                                                                                                                          |
| Venue & Year   | ICCV, 2025                                                                                                                             |
| Key Technique  | Utilizes 3D Gaussian Splatting (3DGS) to reconstruct drone video into a top-down Bird’s Eye View (BEV) projection.                    |
| Metrics        | Outperforms conventional video-based methods on the UniV dataset, demonstrating extreme robustness to lower elevations and occlusions. |
| Relevance      | Physically alters the input data to remove the perspective gap before the neural network processes the image.                          |

While CVCAM handles alignment in the latent space, geometric pre-processing can directly simplify the network's processing burden. Historically, polar and spherical transformations were utilized to unroll images, but these introduced severe blurring, spatial distortion, and loss of fine-grained details.34

The Video2BEV paradigm (2025) leverages the multi-view nature of drone flight to completely bypass single-image limitations.34 It utilizes 3D Gaussian Splatting (3DGS) to construct a rich, continuous 3D representation from a short sequence of drone video frames.34 This 3D representation is then projected orthographically into a Bird’s Eye View (BEV) plane.34 The impact of this transformation is profound: BEV projections fundamentally eliminate the perspective mismatch, aligning the drone's viewpoint identically to the satellite's top-down orientation. Furthermore, because it reconstructs from video, the 3DGS process recovers ground regions that were occluded by buildings or trees in any single frame, drastically improving matching robustness in dense, occluded urban environments.34

## ---

5. Altitude and Scale-Aware Methodologies

The core defining challenge of the SUES-200 benchmark is the necessity to match across drone altitudes ranging continuously from 150m to 300m.1 Most retrieval methods fail because they operate under the assumption of scale consistency, treating a building captured at 150m identically to one captured at 300m, leading to severe field-of-view misalignment.4

### 5.1. Semantic Anchoring for Absolute Metric Scale

| Paper Metadata | Details                                                                                                                                        |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Title          | Scale-Aware Cross-View Geo-Localization via Semantic Anchors and Decoupled Stereoscopic Projection                                             |
| Authors        | (Preprint Authors)                                                                                                                             |
| Venue & Year   | arXiv, March 2026                                                                                                                              |
| Key Technique  | Exploits small vehicles (SVs) as metric references to estimate absolute image scale via a Decoupled Stereoscopic Projection Model.             |
| Metrics        | Significantly improves CVGL robustness under unknown UAV image scales on augmented DenseUAV and UAV-VisLoc datasets.                           |
| Relevance      | Allows the baseline model to dynamically crop the satellite gallery to match the exact field-of-view of the drone prior to feature extraction. |

A highly novel approach specifically tackles scale ambiguity by recovering the absolute metric scale directly from monocular UAV images.4 This is achieved by utilizing small passenger vehicles (SVs) as "semantic anchors." Because passenger vehicles have a stable, universally known prior size distribution, a lightweight object detector isolates them within the drone view.

A Decoupled Stereoscopic Projection model decomposes these 2D vehicle detections into radial and tangential dimensions, mathematically compensating for the perspective distortions inherent in viewing 3D vehicles from oblique angles.4 By applying an Interquartile Range (IQR) robust aggregation to filter detection noise, the algorithm infers the absolute altitude and global scale of the image.4 This scale factor is subsequently injected into the system as a physical constraint. It guides a scale-adaptive cropping mechanism on the satellite gallery, guaranteeing that the satellite patch fed into the ConvNeXt baseline matches the exact spatial boundaries of the drone's field-of-view, entirely neutralizing the SUES-200 altitude variations.4

### 5.2. Multi-Scale Feature Fusion via MSCM

At the architectural level, variations in altitude necessitate multi-scale processing. As demonstrated by the highly successful MCFA model (discussed in Section 1), integrating a Multi-Scale Cascade Module (MSCM) allows the network to aggregate contextual regions of varying receptive fields.12 Rather than relying on a single pooled representation, MSCM dynamically associates local region information, allowing the model to adaptively select the feature scale that best correlates with the satellite view. This ensures high feature stability across all SUES-200 altitude brackets, preventing the catastrophic drops in accuracy typically seen when evaluating models trained at 150m on 300m test data.12

## ---

6. Efficient Inference for Edge Deployment

Deploying complex cross-view retrieval pipelines and VLM-distilled architectures on UAVs requires overcoming stringent Size, Weight, and Power (SWaP) constraints. While the ConvNeXt-Tiny baseline is relatively efficient, next-generation deployment relies on Quantization-Aware Training (QAT) and extreme parameter compression.

### 6.1. TAT-VPR: Ternary Adaptive Transformers (2024)

| Paper Metadata | Details                                                                                                           |
| -------------- | ----------------------------------------------------------------------------------------------------------------- |
| Title          | TAT-VPR: Ternary Adaptive Transformer for Dynamic and Efficient Visual Place Recognition                          |
| Authors        | (ResearchGate Authors)                                                                                            |
| Venue & Year   | IEEE T PATTERN ANAL / ResearchGate, 2024                                                                          |
| Key Technique  | Fuses ternary weights (![]()) with a learned activation-sparsity gate for dynamic compute control.                  |
| Metrics        | Controls computation by up to 40% at run-time without degrading the baseline Recall@1 performance.                |
| Relevance      | Proves that transformers and hierarchical CNNs can be heavily quantized without losing spatial matching fidelity. |

TAT-VPR achieves massive computational reductions by transitioning the architecture to a ternary-quantized state, where weights are constrained to the values ![]().36 It goes a step further by fusing these ternary weights with a learned activation-sparsity gate.36 This gate evaluates the significance of spatial tokens on the fly, allowing the model to dynamically bypass inactive or low-information tokens (e.g., solid patches of sky or flat grass).36 This methodology enables real-time control of compute complexity, dropping FLOPs by up to 40% dynamically based on the available power budget of the micro-UAV, all without degrading the baseline R@1 score.36

### 6.2. TeTRA-VPR: Progressive 2-Bit Distillation (2025)

| Paper Metadata | Details                                                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Title          | TeTRA-VPR: A Ternary Transformer Approach for Compact Visual Place Recognition                                            |
| Authors        | O. Grainge, M. Milford, et al.                                                                                            |
| Venue & Year   | IEEE Robotics and Automation Letters, 2025                                                                                |
| Key Technique  | Progressive quantization strategy mapping a full-precision teacher to a 2-bit backbone with binarized final embeddings.   |
| Metrics        | Reduces memory consumption by 69% and inference latency by 35% with no loss in recall accuracy.                           |
| Relevance      | Allows the vast gallery search process to transition from slow cosine similarity to ultra-fast hardware Hamming distance. |

Taking quantization to its theoretical limit, TeTRA-VPR completely binarizes the final embedding layer while utilizing a 2-bit (ternary) backbone.11

Extreme quantization typically causes severe convergence instability and catastrophic representation collapse, destroying the nuanced spatial features required for geo-localization.38 TeTRA solves this via a carefully designed progressive distillation strategy. It incrementally refines both ternary and binary representations, smoothing the transition from a full-precision teacher (e.g., the SigLIP2-distilled ConvNeXt) to the quantized student.11

The most profound impact of TeTRA-VPR is on retrieval latency. By quantizing the final descriptor to a discrete binary sequence, the similarity computation transitions from floating-point cosine distance to Hamming distance.37 Hamming distance is executed via simple bitwise XOR and POPCOUNT operations directly at the CPU/GPU hardware level, accelerating the nearest-neighbor search across the massive satellite gallery by an order of magnitude.11 This process results in a 69% reduction in memory consumption and a 35% drop in inference latency while matching the accuracy of uncompressed architectures.11

## ---

7. Advanced Losses, Data Augmentation, and Re-Ranking

The final vector of performance maximization occurs post-feature extraction and during loss formulation. The transition from an 85% baseline to a >95% SOTA is frequently dictated by how the network resolves "hard negatives"—distinct geographic locations with practically identical visual topologies.

### 7.1. Advanced Loss Formulation

While the InfoNCE (used in models like Sample4Geo) and standard Triplet Margin losses are foundational, they often suffer from gradient saturation when positive and negative pairs become visually similar.5 Integrating advanced margin-based metric learning losses, such as Circle Loss or ArcFace, forces the embedding space to become substantially more discriminative. ArcFace introduces an additive angular margin penalty to the target logit, simultaneously enhancing intra-class compactness and inter-class discrepancy on a hypersphere. Circle Loss goes further by offering a unified perspective on class-level and pair-wise labels, dynamically re-weighting the gradients of positive and negative similarities based on their distance from the optimum. This ensures that the model does not waste training cycles on easy negatives, focusing entirely on separating identical-looking buildings located in different geographic coordinates.

Furthermore, integrating Supervised Contrastive Learning (SupContrast) allows the model to leverage multiple positive views (e.g., drone shots from different angles or augmented weather variants) simultaneously, pulling all semantically identical representations together in the latent space far more effectively than standard pairwise triplet loss.

### 7.2. GeoVLM: Vision-Language Re-Ranking (2025)

| Paper Metadata | Details                                                                                                     |
| -------------- | ----------------------------------------------------------------------------------------------------------- |
| Title          | GeoVLM: Improving Automated Vehicle Geolocalisation Using Vision-Language Matching                          |
| Authors        | B. Dagda, M. Awais, S. Fallah                                                                               |
| Venue & Year   | arXiv / Robotics (cs.RO), May 2025                                                                          |
| Key Technique  | Trainable re-ranking module utilizing the zero-shot reasoning capabilities of large Vision-Language Models. |
| Metrics        | 93.15% R@1 and 95.23% AP in Drone2Sat settings, outperforming Sample4Geo.                                   |
| Relevance      | Solves the "confusion gallery" problem by logically verifying structural topology using textual reasoning.  |

Standard Euclidean or Cosine distance metrics fail when pixel statistics align but geographic context does not. GeoVLM introduces a secondary, trainable re-ranking module powered by a zero-shot Vision-Language Model to resolve top-k ambiguity.41

Following the coarse retrieval stage (where the ConvNeXt baseline identifies the top-5 or top-10 candidate satellite images), the query and each candidate are processed by the VLM.42 The VLM is explicitly prompted to generate interpretable cross-view language descriptions, logically verifying the structural layout, road topology, and semantic landmarks.41 For example, the VLM assesses: "The drone view shows a T-intersection with a red building on the left and a parking lot on the right; does candidate satellite image #2 reflect this exact topological layout?".41

The text embeddings representing this spatial reasoning are projected into a shared latent space alongside the visual embeddings. A cross-embedding aligner, composed of fully connected layers, processes the summation and outputs a refined ranking score via a sigmoid activation.43 This multi-modal re-ranking drastically reduces false-positive matches in homogeneous urban environments, pushing Top-1 match accuracy to SOTA levels.41

### 7.3. Cross-View Specific Data Augmentation

To build invariance to the drastic environmental changes seen in SUES-200, specialized data augmentation is critical.

* WeatherPrompt and Degradation Simulation: Drawing from the AGEN architecture, applying extreme weather simulations (heavy rain, snow, fog) to the drone imagery during training forces the network to rely on geometric structure rather than transient textural clues or illumination.5
* Rotational and Orthogonal Augmentation: To ensure the model is perfectly invariant to the drone's yaw angle, continuous 360-degree rotational augmentations on the drone view, paired with orthogonal padding/cropping, ensures the CVCAM/MHSAM attention heads learn omnidirectional spatial features rather than relying on standard north-aligned grid biases.45

## ---

8. Conclusion and Strategic Implementation Roadmap

To elevate the baseline ConvNeXt-Tiny + DINOv2 model from 82.35% to the upper echelons of the SUES-200 benchmark (>95% R@1), an integration of the 2024–2025 literature dictates a multi-stage architectural overhaul. The strategic implementation roadmap is as follows:

1. VLM Distillation and Feature Grounding: Maintain the highly efficient ConvNeXt-Tiny backbone, but upgrade the distillation framework from standard DINOv2 to the Asymmetric Mixture of Experts (AMoE). Utilize DINOv3 for dense geometry and SigLIP2 for zero-shot text-aligned spatial localization. Implement Asymmetric Relation Knowledge Distillation (ARKD) to enforce exact patch-level relational consistency between the ConvNeXt student and the massive SigLIP2 teacher.
2. Object-Centric Representation: Integrate Disentangled Slot Attention (DISA) into the representation head. By explicitly partitioning the latent space into independent subsets for shape and texture, the model will naturally bypass the severe domain gap between drone-view facades and satellite-view roofs, comparing images solely on geometric validity.
3. Viewpoint and Scale Alignment: Deploy the CVCAM and MHSAM modules to replace standard global average pooling, allowing the model to iteratively cross-attend to context and filter irrelevant edge noise. Concurrently, utilize semantic anchoring (via passenger vehicle detection) to determine absolute metric scale, allowing the system to dynamically crop the satellite gallery and neutralize the 150m-to-300m altitude variations defining the SUES-200 dataset.
4. Loss Formulation and Re-Ranking: Transition from basic InfoNCE to Circle Loss or ArcFace to penalize hard negatives in the confusion gallery mathematically. For final deployment, pass the Top-5 retrieved candidates through a lightweight, frozen GeoVLM reasoning module to logically and textually verify structural correspondence.
5. Edge-Optimized Execution: Finally, apply the TeTRA-VPR progressive distillation methodology to convert the ultimate embedding space into a binary format. This will allow for instantaneous, hardware-accelerated Hamming-distance gallery retrievals onboard the UAV, cutting memory constraints by 69% without sacrificing the profound representational capacity built in the preceding steps.

By systematically adopting these contemporary methodologies, the geo-localization pipeline will transform from a standard metric-learning feature extractor into an explicitly grounded, geometrically aware, and highly efficient spatial reasoning system capable of dominating the SUES-200 benchmark.

#### Nguồn trích dẫn

1. SUES-200: A Multi-height Multi-scene Cross-view Image Benchmark Across Drone and Satellite | Request PDF - ResearchGate, truy cập vào tháng 3 10, 2026, [https://www.researchgate.net/publication/368872700_SUES-200_A_Multi-height_Multi-scene_Cross-view_Image_Benchmark_Across_Drone_and_Satellite](https://www.researchgate.net/publication/368872700_SUES-200_A_Multi-height_Multi-scene_Cross-view_Image_Benchmark_Across_Drone_and_Satellite)
2. [2204.10704] SUES-200: A Multi-height Multi-scene Cross-view Image Benchmark Across Drone and Satellite - arXiv, truy cập vào tháng 3 10, 2026, [https://arxiv.org/abs/2204.10704](https://arxiv.org/abs/2204.10704)
3. A Dual Attention Approach with Cross-view Interaction and Multi-Scale Spatial Features, truy cập vào tháng 3 10, 2026, [https://arxiv.org/html/2510.27139v1](https://arxiv.org/html/2510.27139v1)
4. [2603.07535] Scale-Aware UAV-to-Satellite Cross-View Geo-Localization: A Semantic Geometric Approach - arXiv, truy cập vào tháng 3 10, 2026, [https://arxiv.org/abs/2603.07535](https://arxiv.org/abs/2603.07535)
5. AGEN: Adaptive Error Control-Driven Cross-View Geo-Localization ..., truy cập vào tháng 3 10, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12196939/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12196939/)
6. Aerial examples for University-1652, CVUSA and CVACT. Polar transform... - ResearchGate, truy cập vào tháng 3 10, 2026, [https://www.researchgate.net/figure/Aerial-examples-for-University-1652-CVUSA-and-CVACT-Polar-transform-is-only-applied-to_fig4_360817202](https://www.researchgate.net/figure/Aerial-examples-for-University-1652-CVUSA-and-CVACT-Polar-transform-is-only-applied-to_fig4_360817202)
7. MobileGeo: Exploring Hierarchical Knowledge Distillation for Resource-Efficient Cross-view Drone Geo-Localization - arXiv, truy cập vào tháng 3 10, 2026, [https://arxiv.org/html/2510.22582v3](https://arxiv.org/html/2510.22582v3)
8. Identifiable Object-Centric Representation Learning via Probabilistic Slot Attention - arXiv.org, truy cập vào tháng 3 10, 2026, [https://arxiv.org/pdf/2406.07141?](https://arxiv.org/pdf/2406.07141)
9. Improving Cross-view Object Geo-localization: A Dual Attention ..., truy cập vào tháng 3 10, 2026, [https://arxiv.org/abs/2510.27139](https://arxiv.org/abs/2510.27139)
10. SigLIP 2: Multilingual Vision-Language Encoders with ... - arXiv, truy cập vào tháng 3 10, 2026, [https://arxiv.org/abs/2502.14786](https://arxiv.org/abs/2502.14786)
11. TeTRA-VPR: A Ternary Transformer Approach for Compact Visual Place Recognition | Request PDF - ResearchGate, truy cập vào tháng 3 10, 2026, [https://www.researchgate.net/publication/393473248_TeTRA-VPR_A_Ternary_Transformer_Approach_for_Compact_Visual_Place_Recognition](https://www.researchgate.net/publication/393473248_TeTRA-VPR_A_Ternary_Transformer_Approach_for_Compact_Visual_Place_Recognition)
12. MCFA: Multi-Scale Cascade and Feature Adaptive Alignment Network for Cross-View Geo-Localization - MDPI, truy cập vào tháng 3 10, 2026, [https://www.mdpi.com/1424-8220/25/14/4519](https://www.mdpi.com/1424-8220/25/14/4519)
13. UAV-Satellite Cross-View Image Matching Based on Adaptive Threshold-Guided Ring Partitioning Framework - ResearchGate, truy cập vào tháng 3 10, 2026, [https://www.researchgate.net/publication/393730508_UAV-Satellite_Cross-View_Image_Matching_Based_on_Adaptive_Threshold-Guided_Ring_Partitioning_Framework](https://www.researchgate.net/publication/393730508_UAV-Satellite_Cross-View_Image_Matching_Based_on_Adaptive_Threshold-Guided_Ring_Partitioning_Framework)
14. UAV-Satellite Cross-View Image Matching Based on Adaptive Threshold-Guided Ring Partitioning Framework - MDPI, truy cập vào tháng 3 10, 2026, [https://www.mdpi.com/2072-4292/17/14/2448](https://www.mdpi.com/2072-4292/17/14/2448)
15. MCFA: Multi-Scale Cascade and Feature Adaptive Alignment Network for Cross-View Geo-Localization - PMC, truy cập vào tháng 3 10, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12299452/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12299452/)
16. MobileGeo: Exploring Hierarchical Knowledge Distillation for Resource-Efficient Cross-view Drone Geo-Localization - arXiv, truy cập vào tháng 3 10, 2026, [https://arxiv.org/html/2510.22582v2](https://arxiv.org/html/2510.22582v2)
17. Cross-View UAV Geo-Localization with Precision-Focused Efficient Design: A Hierarchical Distillation Approach with Multi-view Refinement - arXiv, truy cập vào tháng 3 10, 2026, [https://arxiv.org/html/2510.22582v1](https://arxiv.org/html/2510.22582v1)
18. Object-Centric Slot Disentanglement - Emergent Mind, truy cập vào tháng 3 10, 2026, [https://www.emergentmind.com/topics/object-centric-slot-disentanglement](https://www.emergentmind.com/topics/object-centric-slot-disentanglement)
19. Slot-Based Representation - Emergent Mind, truy cập vào tháng 3 10, 2026, [https://www.emergentmind.com/topics/slot-based-representation](https://www.emergentmind.com/topics/slot-based-representation)
20. Correspondence-Aware Masked Attention - Emergent Mind, truy cập vào tháng 3 10, 2026, [https://www.emergentmind.com/topics/correspondence-aware-masked-attention-module](https://www.emergentmind.com/topics/correspondence-aware-masked-attention-module)
21. Slot-Guided Adaptation of Pre-trained Diffusion Models for Object-Centric Learning and Compositional Generation | OpenReview, truy cập vào tháng 3 10, 2026, [https://openreview.net/forum?id=kZvor5aaz7](https://openreview.net/forum?id=kZvor5aaz7)
22. Explicitly Disentangled Representations in Object-Centric Learning | OpenReview, truy cập vào tháng 3 10, 2026, [https://openreview.net/forum?id=r8UFp9olQ0](https://openreview.net/forum?id=r8UFp9olQ0)
23. Learning Global Object-Centric Representations via Disentangled Slot Attention | Request PDF - ResearchGate, truy cập vào tháng 3 10, 2026, [https://www.researchgate.net/publication/385215805_Learning_Global_Object-Centric_Representations_via_Disentangled_Slot_Attention](https://www.researchgate.net/publication/385215805_Learning_Global_Object-Centric_Representations_via_Disentangled_Slot_Attention)
24. Object-Aware DINO (Oh-A-Dino): Enhancing Self-Supervised Representations for Multi-Object Instance Retrieval - arXiv.org, truy cập vào tháng 3 10, 2026, [https://arxiv.org/html/2503.09867v2](https://arxiv.org/html/2503.09867v2)
25. Object-Aware DINO (Oh-A-Dino): Enhancing Self-Supervised Representations for Multi-Object Instance Retrieval - arXiv, truy cập vào tháng 3 10, 2026, [https://arxiv.org/html/2503.09867v1](https://arxiv.org/html/2503.09867v1)
26. QASA: Quality-Guided K-Adaptive Slot Attention for Unsupervised Object-Centric Learning, truy cập vào tháng 3 10, 2026, [https://arxiv.org/html/2601.12936v1](https://arxiv.org/html/2601.12936v1)
27. Adaptive Slot Attention: Object Discovery with Dynamic Slot Number - Semantic Scholar, truy cập vào tháng 3 10, 2026, [https://www.semanticscholar.org/paper/de9402a5e259f2c5eb919914bc898a3ca9eccdef](https://www.semanticscholar.org/paper/de9402a5e259f2c5eb919914bc898a3ca9eccdef)
28. SigLIP2: Dual-Tower Multilingual Vision-Language Encoders - Emergent Mind, truy cập vào tháng 3 10, 2026, [https://www.emergentmind.com/topics/siglip2-model](https://www.emergentmind.com/topics/siglip2-model)
29. THE SPATIAL BLINDSPOT OF VISION-LANGUAGE MODELS - OpenReview, truy cập vào tháng 3 10, 2026, [https://openreview.net/pdf?id=ZTftkiU3Hd](https://openreview.net/pdf?id=ZTftkiU3Hd)
30. AMoE: Agglomerative Mixture-of-Experts Vision Foundation Model - arXiv, truy cập vào tháng 3 10, 2026, [https://arxiv.org/html/2512.20157v1](https://arxiv.org/html/2512.20157v1)
31. Vision Encoders in Vision-Language Models: A Survey - Jina AI, truy cập vào tháng 3 10, 2026, [https://jina.ai/vision-encoder-survey.pdf](https://jina.ai/vision-encoder-survey.pdf)
32. Daily Papers - Hugging Face, truy cập vào tháng 3 10, 2026, [https://huggingface.co/papers?q=computation-efficient%20image%20encoder](https://huggingface.co/papers?q=computation-efficient+image+encoder)
33. Improving Cross-view Object Geo-localization - arXiv.org, truy cập vào tháng 3 10, 2026, [https://arxiv.org/pdf/2510.27139](https://arxiv.org/pdf/2510.27139)
34. Video2BEV: Transforming Drone Videos to ... - CVF Open Access, truy cập vào tháng 3 10, 2026, [https://openaccess.thecvf.com/content/ICCV2025/papers/Ju_Video2BEV_Transforming_Drone_Videos_to_BEVs_for_Video-based_Geo-localization_ICCV_2025_paper.pdf](https://openaccess.thecvf.com/content/ICCV2025/papers/Ju_Video2BEV_Transforming_Drone_Videos_to_BEVs_for_Video-based_Geo-localization_ICCV_2025_paper.pdf)
35. Video2BEV: Transforming Drone Videos to BEVs for Video-based Geo-localization - arXiv, truy cập vào tháng 3 10, 2026, [https://arxiv.org/html/2411.13610v4](https://arxiv.org/html/2411.13610v4)
36. Design Space Exploration of Low-Bit Quantized Neural Networks for Visual Place Recognition | Request PDF - ResearchGate, truy cập vào tháng 3 10, 2026, [https://www.researchgate.net/publication/379679766_Design_Space_Exploration_of_Low-Bit_Quantized_Neural_Networks_for_Visual_Place_Recognition](https://www.researchgate.net/publication/379679766_Design_Space_Exploration_of_Low-Bit_Quantized_Neural_Networks_for_Visual_Place_Recognition)
37. TeTRA-VPR: A Ternary Transformer Approach for Compact Visual Place Recognition - arXiv, truy cập vào tháng 3 10, 2026, [https://arxiv.org/pdf/2503.02511](https://arxiv.org/pdf/2503.02511)
38. TeTRA-VPR: A Ternary Transformer Approach for Compact Visual Place Recognition - arXiv, truy cập vào tháng 3 10, 2026, [https://arxiv.org/html/2503.02511v1](https://arxiv.org/html/2503.02511v1)
39. (PDF) TeTRA-VPR: A Ternary Transformer Approach for Compact Visual Place Recognition - ResearchGate, truy cập vào tháng 3 10, 2026, [https://www.researchgate.net/publication/389581008_TeTRA-VPR_A_Ternary_Transformer_Approach_for_Compact_Visual_Place_Recognition](https://www.researchgate.net/publication/389581008_TeTRA-VPR_A_Ternary_Transformer_Approach_for_Compact_Visual_Place_Recognition)
40. Object Detection as an Optional Basis: A Graph Matching Network for Cross-View UAV Localization - arXiv.org, truy cập vào tháng 3 10, 2026, [https://arxiv.org/html/2511.02489v1](https://arxiv.org/html/2511.02489v1)
41. GeoVLM: Improving Automated Vehicle Geolocalisation Using Vision-Language Matching, truy cập vào tháng 3 10, 2026, [https://arxiv.org/html/2505.13669v1](https://arxiv.org/html/2505.13669v1)
42. VICI: VLM-Instructed Cross-view Image-localisation, truy cập vào tháng 3 10, 2026, [https://personalpages.surrey.ac.uk/s.hadfield/papers/Shore25c.pdf](https://personalpages.surrey.ac.uk/s.hadfield/papers/Shore25c.pdf)
43. [Literature Review] GeoVLM: Improving Automated Vehicle Geolocalisation Using Vision-Language Matching - Moonlight, truy cập vào tháng 3 10, 2026, [https://www.themoonlight.io/en/review/geovlm-improving-automated-vehicle-geolocalisation-using-vision-language-matching](https://www.themoonlight.io/en/review/geovlm-improving-automated-vehicle-geolocalisation-using-vision-language-matching)
44. MuseNet/State-of-the-art.md at master - GitHub, truy cập vào tháng 3 10, 2026, [https://github.com/wtyhub/MuseNet/blob/master/State-of-the-art.md](https://github.com/wtyhub/MuseNet/blob/master/State-of-the-art.md)
45. The specific image cases on SUES-200 [22], where (a)satellite-view... - ResearchGate, truy cập vào tháng 3 10, 2026, [https://www.researchgate.net/figure/The-specific-image-cases-on-SUES-20022-where-asatellite-view-image-and-b-c_fig7_383699869](https://www.researchgate.net/figure/The-specific-image-cases-on-SUES-20022-where-asatellite-view-image-and-b-c_fig7_383699869)
46. 计算机视觉与模式识别2025_10_14, truy cập vào tháng 3 10, 2026, [https://www.arxivdaily.com/thread/72691](https://www.arxivdaily.com/thread/72691)

**
