# Project architecture & Tensor Flow Map
This document traces the logical data flow (input/output shape) for each project pipeline.

Legend:\
*B = Batch Size,\
C = Channels,\
H = Height,\
W = Width,\
N = Number of Objects/Queries.*

---
## Centralized baseline 0.1: Commercial Faster R-CNN (Off-the-shelf)
**Objective:** Establish the lower limit and prove the presence of the *domain gap* between generic data (COCO commercial model) and ZOD.

**Pipeline:**
- Backbone: ResNet-50 (default public COCO backbone)
- Head: Faster R-CNN (initialized with COCO pubilc weights) --> fine-tuning on ZOD


## Centralized baseline 0.2: ResNet-50 + Faster R-CNN
**Objective:** Evaluation of a classic CNN architecture pre-trained on ZOD. This will serve as a pretty decent baseline.

**Pipeline:**
- *Backbone:* ResNet-50 initialized with DINOv1 public weights --> fine-tuning on ZOD (100 epochs, almost 1.5 days)
- *Object detection head:* Faster R-CNN (the backbone is the trained ResNet-50, the head is initialized with COCO public weights) --> fine-tuning on ZOD with a "end-to-end fine tuning" approach (NON frozen backbone)


### Tensor Flow
1. **Dataloader (ZOD):**
   - Image Input: `[B, 3, 1024, 1024]`
   - Target Input (Ground Truth): List of dictionaries with `boxes` `[N_real, 4]` and `labels` `[N_real]`
2. **Backbone (ResNet-50):**
   - Extracts spatial feature maps.
   - Feature Map Output: `[B, 2048, 32, 32]` *(varies based on the extracted layer)*
3. **Detection Head (Faster R-CNN):**
   - Generates proposals and classifies bounding boxes.
   - Prediction Output: List of dictionaries with `boxes` `[N_pred, 4]`, `labels` `[N_pred]`, `scores` `[N_pred]`
4. **Evaluation Module:**
   - Compares predicted vs. real `boxes` by calculating IoU and generates the **mAP**.

---

<!-- ## 2. Baseline 1: ViT (DINOv3) + plainDETR (Centralized)
**Objective:** Evaluation of SOTA Vision Transformer in a centralized environment. Head initialized from scratch.
**Status:** To be implemented.

### Tensor Flow
1. **Dataloader (ZOD):**
   - Image Input: `[B, 3, 1024, 1024]`
   - Target Input: Dictionaries with `boxes` and `labels`
2. **Backbone (Vision Transformer - DINOv3 pre-trained weights):**
   - The image is split into patches and flattened into a sequence.
   - Embedding Output: `[B, Num_Patches, Embed_Dim]` *(e.g., [B, 4096, 768])*
3. **Detection Head (plainDETR - random weights):**
   - The Transformer decoder uses "Object Queries" to locate objects.
   - Bounding Box Prediction Output: `[B, Num_Queries, 4]`
   - Class Logits Prediction Output: `[B, Num_Queries, Num_Classes + 1]`

---

## 3. FSSL Pipeline: ViT + plainDETR (Federated - Partitioned ZOD)
**Objective:** Test the impact of Non-IID partitioning on the DINOv3 backbone.
**Status:** In development (Step 3).

### Architectural Flow
- **Client (x K):** Each has a local Dataloader (ZOD filtered by metadata: e.g., `vehicle_id`).
- **Local Training:** Performs fine-tuning/continuous learning of the ViT backbone using contrastive/distillation loss.
- **Aggregation (Server):** Receives weight tensors `[Client_K_ViT_Parameters]` and applies FedAvg / FedCA.
- **Evaluation:** The aggregated global model is passed to the "Baseline 1" pipeline to calculate the performance drop. -->