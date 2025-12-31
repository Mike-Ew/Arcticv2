# Data Profile Verification (Generated)

Generated: 2025-12-31 01:50:29

## Executive Summary
- Nearly half of all pixels are ignored (255), so effective supervision is much smaller than raw pixel counts.
- Validation is strongly OpenWater‑skewed and lacks ThinFirstYearIce crops, making rare‑class validation metrics unstable.
- Summer months dominate crop volume but contribute almost no rare classes, amplifying imbalance unless sampling is adjusted.
- Region split is strict (only 3 regions in val), which is good for generalization tests but increases distribution shift.
- Current coords define the effective dataset; extra memmap scenes exist but are unused in training.

## Sources
- data_dir: `/home/mike1/Arcticv2/data/ai4arctic_hugging face`
- memmap_dir: `/home/mike1/Arcticv2/data/ai4arctic_hugging face/npy_memmap`
- metadata.csv: `/home/mike1/Arcticv2/data/ai4arctic_hugging face/metadata.csv`
- coords: `train_coords.npy`, `val_coords.npy`

## Scene Inventory
| Metric | Value |
|---|---|
| metadata.csv scenes | 532 |
| metadata train scenes | 512 |
| metadata test scenes | 20 |
| memmap train scenes (files) | 505 |
| memmap val scenes (files) | 137 |
| coords train scenes | 445 |
| coords val scenes | 67 |

### Scene Mismatches
- memmap scenes missing in metadata: 0
- metadata scenes missing in memmap: 20
  - example: ['20180124T194759_dmi_prep', '20180623T114935_cis_prep', '20180707T113313_cis_prep', '20180716T110418_cis_prep', '20180903T123331_cis_prep']

## Split Summary
| Split | Scenes | Crops |
|---|---|---|
| train | 505 | 84652 |
| val | 137 | 14264 |

## Label Integrity
### Train
- invalid label pixels (not in 0-5 or 255): 0
- ignore (255) pixels: 6297453105 (48.78%)

### Val
- invalid label pixels (not in 0-5 or 255): 0
- ignore (255) pixels: 1638687141 (47.21%)

### Combined
- invalid label pixels (not in 0-5 or 255): 0
- ignore (255) pixels: 7936140246 (48.45%)

## Class Distribution (Pixels)
### Train
| Class | Name | Pixels | Frequency |
|---|---|---|---|
| 0 | OpenWater | 4,358,716,505 | 65.92% |
| 1 | NewIce | 116,907,458 | 1.77% |
| 2 | YoungIce | 168,799,195 | 2.55% |
| 3 | ThinFirstYearIce | 143,306,231 | 2.17% |
| 4 | ThickFirstYearIce | 1,000,555,483 | 15.13% |
| 5 | OldIce | 823,401,660 | 12.45% |

### Val
| Class | Name | Pixels | Frequency |
|---|---|---|---|
| 0 | OpenWater | 1,414,406,697 | 77.20% |
| 1 | NewIce | 30,919,223 | 1.69% |
| 2 | YoungIce | 28,786,638 | 1.57% |
| 3 | ThinFirstYearIce | 16,856,043 | 0.92% |
| 4 | ThickFirstYearIce | 182,571,152 | 9.96% |
| 5 | OldIce | 158,613,748 | 8.66% |

### Combined
| Class | Name | Pixels | Frequency |
|---|---|---|---|
| 0 | OpenWater | 5,773,123,202 | 68.37% |
| 1 | NewIce | 147,826,681 | 1.75% |
| 2 | YoungIce | 197,585,833 | 2.34% |
| 3 | ThinFirstYearIce | 160,162,274 | 1.90% |
| 4 | ThickFirstYearIce | 1,183,126,635 | 14.01% |
| 5 | OldIce | 982,015,408 | 11.63% |

## Scene Presence by Class
### Train
| Class | Name | Scenes w/ class | Percent of scenes |
|---|---|---|---|
| 0 | OpenWater | 437 | 86.53% |
| 1 | NewIce | 141 | 27.92% |
| 2 | YoungIce | 119 | 23.56% |
| 3 | ThinFirstYearIce | 108 | 21.39% |
| 4 | ThickFirstYearIce | 238 | 47.13% |
| 5 | OldIce | 170 | 33.66% |

### Val
| Class | Name | Scenes w/ class | Percent of scenes |
|---|---|---|---|
| 0 | OpenWater | 127 | 92.70% |
| 1 | NewIce | 33 | 24.09% |
| 2 | YoungIce | 33 | 24.09% |
| 3 | ThinFirstYearIce | 16 | 11.68% |
| 4 | ThickFirstYearIce | 50 | 36.50% |
| 5 | OldIce | 52 | 37.96% |

## Crop Presence by Class (from coords)
### Train
| Class | Name | Crops w/ class | Percent of crops |
|---|---|---|---|
| 0 | OpenWater | 56843 | 67.15% |
| 1 | NewIce | 3821 | 4.51% |
| 2 | YoungIce | 3851 | 4.55% |
| 3 | ThinFirstYearIce | 3470 | 4.10% |
| 4 | ThickFirstYearIce | 19869 | 23.47% |
| 5 | OldIce | 14679 | 17.34% |

### Val
| Class | Name | Crops w/ class | Percent of crops |
|---|---|---|---|
| 0 | OpenWater | 13758 | 96.45% |
| 1 | NewIce | 417 | 2.92% |
| 2 | YoungIce | 212 | 1.49% |
| 3 | ThinFirstYearIce | 0 | 0.00% |
| 4 | ThickFirstYearIce | 817 | 5.73% |
| 5 | OldIce | 1016 | 7.12% |

## Month Distribution
### Train
| Month | Scenes | Crops | Rare-class crops |
|---|---|---|---|
| 01 | 35 | 5941 | 2120 |
| 02 | 25 | 3741 | 883 |
| 03 | 30 | 5410 | 660 |
| 04 | 37 | 8249 | 446 |
| 05 | 38 | 7647 | 122 |
| 06 | 31 | 6106 | 1 |
| 07 | 34 | 6719 | 0 |
| 08 | 53 | 10942 | 0 |
| 09 | 54 | 11152 | 691 |
| 10 | 39 | 7577 | 1486 |
| 11 | 34 | 6349 | 1339 |
| 12 | 35 | 4819 | 1820 |

### Val
| Month | Scenes | Crops | Rare-class crops |
|---|---|---|---|
| 01 | 0 | 0 | 0 |
| 02 | 6 | 1745 | 6 |
| 03 | 1 | 238 | 0 |
| 04 | 2 | 456 | 10 |
| 05 | 3 | 931 | 1 |
| 06 | 2 | 368 | 0 |
| 07 | 7 | 1604 | 0 |
| 08 | 12 | 2187 | 0 |
| 09 | 17 | 3163 | 123 |
| 10 | 12 | 2172 | 374 |
| 11 | 3 | 711 | 13 |
| 12 | 2 | 689 | 0 |

## Season Distribution
### Train
| Season | Scenes | Crops |
|---|---|---|
| Winter | 95 | 14501 |
| Spring | 105 | 21306 |
| Summer | 118 | 23767 |
| Fall | 127 | 25078 |
| Unknown | 0 | 0 |

### Val
| Season | Scenes | Crops |
|---|---|---|
| Winter | 8 | 2434 |
| Spring | 6 | 1625 |
| Summer | 21 | 4159 |
| Fall | 32 | 6046 |
| Unknown | 0 | 0 |

## Region Distribution
### Train
| Region | Scenes | Crops | Rare-class crops |
|---|---|---|---|
| SGRDIEA | 42 | 9438 | 1669 |
| CentralEast_RIC | 39 | 7266 | 510 |
| Qaanaaq_RIC | 39 | 6315 | 920 |
| SGRDIFOXE | 39 | 7120 | 1514 |
| SGRDINFLD | 39 | 8087 | 805 |
| NorthEast_RIC | 38 | 7482 | 665 |
| NorthWest_RIC | 37 | 4778 | 540 |
| CentralWest_RIC | 36 | 6412 | 1053 |
| SGRDIWA | 32 | 5244 | 805 |
| North_RIC | 31 | 7551 | 590 |
| SouthWest_RIC | 27 | 6356 | 331 |
| SouthEast_RIC | 26 | 5041 | 93 |
| SGRDIHA | 20 | 3562 | 73 |

### Val
| Region | Scenes | Crops | Rare-class crops |
|---|---|---|---|
| CapeFarewell_RIC | 39 | 10755 | 17 |
| SGRDIMID | 25 | 3367 | 434 |
| NorthAndCentralEast_RIC | 3 | 142 | 76 |

## Ice Service Distribution
### Train
| Service | Scenes | Crops | Rare-class crops |
|---|---|---|---|
| dmi | 273 | 51201 | 4702 |
| cis | 172 | 33451 | 4866 |

### Val
| Service | Scenes | Crops | Rare-class crops |
|---|---|---|---|
| dmi | 42 | 10897 | 93 |
| cis | 25 | 3367 | 434 |

## Notes / Gaps
- Test split is present in metadata, but memmap is only train/val.
- Scene mismatches above indicate memmap files not referenced in metadata or vice versa.
- Coords scene lists (445 train / 67 val) are smaller than memmap scene files (505 / 137), so extra memmap scenes are unused by current dataloaders.
- Crop-level stats depend on `classes` lists stored in coords (preprocess output).

## Analyst Takeaways
- The dataset is heavily masked: ~48% of pixels are 255 (ignore). This is true in train and val and strongly shapes effective supervision.
- Validation is more OpenWater-skewed than training (77% vs 66% of pixels), which can depress rare-class IoU in val even when train improves.
- ThinFirstYearIce is extremely scarce in val at the crop level (0% of crops contain class 3), so per-class validation for that class is unreliable.
- Summer (Jun–Aug) contributes a large share of crops but almost no rare-class crops; without reweighting or sampling control, training will bias toward OpenWater.
- Region split is strict: val contains only 3 regions. This is good for geographic generalization testing but increases distribution shift.
- Extra memmap scenes exist but are unused due to coords lists; current training uses only the coords-defined subset.

## Actionable Implications
### 1) Evaluation Strategy
- Treat per-class metrics for rare classes (especially ThinFirstYearIce) with caution in val; use train rare-class trends + qualitative checks.
- Track class distribution of predictions vs GT; val is OpenWater-heavy so a naive model can look decent on mIoU but fail rare classes.

### 2) Sampling / Weighting
- Prefer seasonal or rare-class oversampling if you need rare-class performance; Jun–Aug can be downweighted without harming rare classes.
- If staying with current loss, consider stronger class weights (10x–25x range for classes 1–3) or focal loss to offset heavy imbalance.

### 3) Split Integrity
- Region split likely increases shift; ensure comparisons are consistent (same split every run) and interpret val as a hard generalization test.
- If you need more stable val metrics, consider a secondary “dev” split (same regions as train) for fast feedback.

### 4) Data Hygiene
- Clean unused memmap scenes if you want strict alignment between memmap and coords; otherwise accept that coords define the effective dataset.

## Suggested Next Steps (Optional)
1. Add a lightweight dev split (same regions as train) for faster iteration and keep region-split val as the final test.
2. Enable rare-class-aware sampling (rare crops 2–3x) or stronger class weights for classes 1–3.
3. For full-scene training, monitor class distribution per scene to ensure the val shift is understood.
