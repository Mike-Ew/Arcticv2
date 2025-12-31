# AI4Arctic Sea Ice Dataset - Comprehensive Data Profile

**Dataset**: AI4Arctic Sea Ice Challenge (Ready-To-Train)
**Source**: [DTU Data Repository](https://data.dtu.dk/articles/dataset/Ready-To-Train_AI4Arctic_Sea_Ice_Challenge_Dataset/21316608)
**Last Updated**: 2025-12-31

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Total Scenes | 532 |
| Training Scenes | 512 (445 after region split) |
| Test Scenes | 20 |
| Training Crops (512x512) | 84,652 |
| Validation Crops (512x512) | 14,264 |
| Total Labeled Pixels | 5.68 billion |
| Scene Size | ~5000 x 5200 pixels @ 80m resolution |
| Physical Coverage per Scene | ~400 km x 416 km |

---

## 2. Input Channels

| Channel | Description | Mean | Std | Notes |
|---------|-------------|------|-----|-------|
| HH | SAR Backscatter (co-pol) | -0.1335 | 0.7164 | **Already in dB** |
| HV | SAR Backscatter (cross-pol) | -0.1923 | 0.4533 | **Already in dB** |
| IncidenceAngle | SAR incidence angle | -0.0791 | 0.6984 | Normalized |

### Critical Note on SAR Data
The SAR data is **already in dB scale** (sigma0 in dB). The negative mean values confirm this.
- **DO NOT** apply additional dB normalization (10*log10)
- This was verified from the official AI4Arctic PDF manual (page 8)

---

## 3. SOD Class Distribution (Stage of Development)

| Class ID | Class Name | Pixels | Frequency | Imbalance | Current Weight |
|----------|------------|--------|-----------|-----------|----------------|
| 0 | OpenWater | 3.51B | 61.79% | 1.0x | 1.00 |
| 1 | NewIce | 104M | 1.82% | **33.9x** | 2.74 |
| 2 | YoungIce | 162M | 2.86% | **21.6x** | 1.75 |
| 3 | ThinFirstYearIce | 143M | 2.52% | **24.5x** | 1.98 |
| 4 | ThickFirstYearIce | 970M | 17.06% | 3.6x | 1.00 |
| 5 | OldIce | 793M | 13.95% | 4.4x | 1.00 |

### Visual Class Imbalance
```
OpenWater           : ############################################################ (61.8%)
NewIce              : #                                                            (1.8%)
YoungIce            : ##                                                           (2.9%)
ThinFirstYearIce    : ##                                                           (2.5%)
ThickFirstYearIce   : ################                                             (17.1%)
OldIce              : #############                                                (14.0%)
```

### Key Insight
- **62% of all pixels are OpenWater** - model will easily overfit to this
- **Rare classes (NewIce, YoungIce, ThinFY) are only 7.2% combined**
- Current class weights (max 2.74x) are **too weak** to address this

---

## 4. Temporal Distribution

### By Year
| Year | Scenes | Percentage |
|------|--------|------------|
| 2018 | 88 | 17.2% |
| 2019 | 128 | 25.0% |
| 2020 | 117 | 22.9% |
| 2021 | 179 | 35.0% |

### By Month
```
Jan: 35  #######
Feb: 31  ######
Mar: 31  ######
Apr: 39  #######
May: 41  ########
Jun: 33  ######
Jul: 41  ########
Aug: 65  ############# (Peak - mostly OpenWater)
Sep: 71  ############## (Peak - ice forming)
Oct: 51  ##########
Nov: 37  #######
Dec: 37  #######
```

### By Season
| Season | Scenes | Percentage | Rare Class Presence |
|--------|--------|------------|---------------------|
| Winter (Dec-Feb) | 103 | 20.1% | HIGH - YoungIce, ThinFY |
| Spring (Mar-May) | 111 | 21.7% | MEDIUM |
| Summer (Jun-Aug) | 139 | 27.1% | **NONE** - all melted |
| Fall (Sep-Nov) | 159 | 31.1% | HIGH - NewIce forming |

---

## 5. Rare Classes by Month

| Month | Crops | NewIce | YoungIce | ThinFirstYearIce |
|-------|-------|--------|----------|------------------|
| Jan | 5,941 | 158 | **1,308** | **1,102** |
| Feb | 3,741 | 110 | 388 | 620 |
| Mar | 5,410 | 143 | 178 | 485 |
| Apr | 8,249 | 113 | 130 | 267 |
| May | 7,647 | 78 | 15 | 36 |
| Jun | 6,106 | 0 | 0 | 1 |
| Jul | 6,719 | 0 | 0 | 0 |
| Aug | 10,942 | 0 | 0 | 0 |
| Sep | 11,152 | 653 | 91 | 0 |
| Oct | 7,577 | **1,289** | 431 | 32 |
| Nov | 6,349 | 784 | 542 | 144 |
| Dec | 4,819 | 493 | **768** | **783** |

### Best Months for Rare Classes
- **NewIce**: Oct (1,289), Nov (784), Sep (653) - Fall ice formation
- **YoungIce**: Jan (1,308), Dec (768), Nov (542) - Winter thickening
- **ThinFirstYearIce**: Jan (1,102), Dec (783), Feb (620) - Winter survival

### Critical Insight
**Summer months (Jun, Jul, Aug) have ZERO rare ice classes!**
- These 139 scenes (27% of data) are almost entirely OpenWater
- Training on these without balancing will hurt rare class performance

---

## 6. Geographic Distribution

### By Ice Service
| Service | Scenes | Percentage | Coverage |
|---------|--------|------------|----------|
| DMI | 315 | 61.5% | Greenland waters |
| CIS | 197 | 38.5% | Canadian Arctic |

### By Region (with rare class counts)
| Region | Crops | NewIce | YoungIce | ThinFirstYearIce |
|--------|-------|--------|----------|------------------|
| SGRDIEA | 9,438 | 540 | **1,169** | 448 |
| SGRDINFLD | 8,087 | 152 | 528 | 334 |
| North_RIC | 7,551 | 225 | 136 | 234 |
| NorthEast_RIC | 7,482 | 270 | 49 | 410 |
| CentralEast_RIC | 7,266 | 333 | 133 | 73 |
| SGRDIFOXE | 7,120 | 408 | 885 | **623** |
| CentralWest_RIC | 6,412 | 306 | 294 | **656** |
| SouthWest_RIC | 6,356 | 165 | 77 | 100 |
| Qaanaaq_RIC | 6,315 | 344 | 186 | 411 |
| **SGRDIWA** | 5,244 | **701** | 229 | 0 |
| SouthEast_RIC | 5,041 | 82 | 7 | 7 |
| NorthWest_RIC | 4,778 | 222 | 158 | 174 |
| SGRDIHA | 3,562 | 73 | 0 | 0 |

### Best Regions for Rare Classes
- **NewIce**: SGRDIWA (701), SGRDIEA (540), SGRDIFOXE (408)
- **YoungIce**: SGRDIEA (1,169), SGRDIFOXE (885), SGRDINFLD (528)
- **ThinFirstYearIce**: CentralWest_RIC (656), SGRDIFOXE (623), SGRDIEA (448)

---

## 7. Recommended Class Weights

### Current Weights (Too Weak)
```python
weights = [1.0, 2.74, 1.75, 1.98, 1.0, 1.0]  # Max only 2.74x
```

### Enhanced Weights (Inverse Frequency)
```python
enhanced_weights = [1.0, 33.87, 21.62, 24.51, 3.62, 4.43]
```

### Practical Recommendation
```python
# Balanced but not extreme
practical_weights = [1.0, 20.0, 15.0, 15.0, 2.5, 3.0]
```

---

## 8. Class Imbalance Mitigation Strategies

### Strategy 1: Enhanced Class-Weighted Loss
```python
class_weights = torch.tensor([1.0, 25.0, 18.0, 20.0, 3.0, 4.0])
criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
```

### Strategy 2: Focal Loss
```python
# Down-weight easy examples (OpenWater)
# gamma=2.0 focuses on hard pixels
from segmentation_models_pytorch.losses import FocalLoss
criterion = FocalLoss(mode='multiclass', gamma=2.0)
```

### Strategy 3: Seasonal Stratified Sampling
```python
# Weight samples by season to ensure rare classes appear
# Winter/Fall crops should be 2x more likely to be sampled
seasonal_weights = {
    'winter': 1.5,  # Dec, Jan, Feb - has YoungIce, ThinFY
    'fall': 1.5,    # Sep, Oct, Nov - has NewIce
    'spring': 1.0,  # Mar, Apr, May
    'summer': 0.5,  # Jun, Jul, Aug - mostly OpenWater
}
```

### Strategy 4: Rare Class Oversampling
```python
# Identify crops with rare classes
rare_class_ids = [1, 2, 3]  # NewIce, YoungIce, ThinFirstYearIce
rare_crop_indices = [
    i for i, classes in enumerate(classes_per_crop)
    if any(c in rare_class_ids for c in classes)
]
# Sample these 3x more frequently
```

### Strategy 5: Region-Aware Batching
```python
# Ensure each batch has crops from rare-class-rich regions
priority_regions = ['SGRDIWA', 'SGRDIEA', 'SGRDIFOXE', 'CentralWest_RIC']
```

---

## 9. Full-Scene Mode (Winner's Recipe)

### Current Implementation
- 10x downsampling: 80m → 800m resolution
- Output size: 512 x 512 pixels
- Physical coverage: ~400 km x 400 km per input
- Uses `cv2.INTER_AREA` for high-quality downsampling

### Why Full-Scene Helps
- At 80m resolution, CNN receptive field is too small
- Ice floes and boundaries span tens of kilometers
- Downsampled full scenes let the model see:
  - Coastlines
  - Open ocean boundaries
  - Ice edge transitions
  - Regional ice patterns

### Performance Impact
| Mode | mIoU (Overfit Test) |
|------|---------------------|
| Crop-based (512x512) | ~0.15 |
| Full-scene (10x downsample) | **0.29** |

---

## 10. Target Metrics

### Competition Benchmarks (MMSeaIce Winners)
| Metric | Winning Score | Our Target (SAR-only) |
|--------|---------------|----------------------|
| SOD F1 | 88.6% | 75-78% |
| SIC R² | 92.0% | 85% |
| Combined | 86.3% | - |

### Per-Class Targets
| Class | Winner Accuracy | Our Target |
|-------|-----------------|------------|
| OpenWater | ~95% | 90%+ |
| NewIce | 19% | 25%+ |
| YoungIce | ~40% | 35%+ |
| ThinFirstYearIce | 31% | 35%+ |
| ThickFirstYearIce | ~70% | 60%+ |
| OldIce | ~60% | 50%+ |

---

## 11. Quick Reference

### Dataset Paths
```python
data_dir = Path("data/ai4arctic_hugging face")
memmap_dir = data_dir / "npy_memmap"
metadata_csv = data_dir / "metadata.csv"
```

### Key Constants
```python
NUM_CLASSES = 6
IGNORE_INDEX = 255  # or -100
CROP_SIZE = 512
SCENE_DOWNSAMPLE = 10
BASE_RESOLUTION = 80  # meters
```

### Critical Reminders
1. **SAR is already in dB** - don't apply log10!
2. **Summer scenes have no rare ice** - balance by season
3. **Current weights are too weak** - increase for rare classes
4. **Full-scene mode doubles performance** - use it!

---

## Citation

If you use this dataset, please cite:

> Buus-Hinkler, Jørgen; Wulf, Tore; Stokholm, Andreas Rønne; Korosov, Anton; Saldo, Roberto; Pedersen, Leif Toudal; et al. (2022). AI4Arctic Sea Ice Challenge Dataset. Technical University of Denmark. Collection. https://doi.org/10.11583/DTU.c.6244065.v2
