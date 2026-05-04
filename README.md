# SCCL: Structural Consistency Contrastive Learning for Partially View-Aligned Multi-View Clustering

## 📌 Overview

Multi-view clustering often suffers from **partial view alignment**, where only a subset of samples have known correspondences across views. Existing methods either overlook view-specific information or fail to exploit unaligned data.

**SCCL** proposes a novel contrastive learning framework that:

- Enforces **semantic consistency** by aligning different views toward shared global cluster centers.
- Introduces a **robust two-level contrastive loss** to propagate alignment knowledge from aligned to unaligned samples.
- Achieves **state-of-the-art performance** on six real-world datasets under both partially aligned and fully aligned settings.
