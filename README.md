# 🚦 Traffic Demand Prediction
### Gridlock Hackathon 2.0 — Flipkart × HackerEarth

![Score](https://img.shields.io/badge/Score-99.74%20%2F%20100-brightgreen)
![R²](https://img.shields.io/badge/Train%20R²-0.9974-blue)
![Model](https://img.shields.io/badge/Model-LightGBM-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)

---

## Problem Statement

Cities worldwide are turning to AI-powered solutions to tackle traffic congestion. The goal is to **predict traffic demand** at a given geolocation and timestamp — providing insights into passenger travel patterns, booking behavior, and trip dynamics in the urban travel industry.

> **Evaluation Metric:** `score = max(0, 100 * metrics.r2_score(actual, predicted))`

---

## Dataset

| File | Shape | Description |
|------|-------|-------------|
| `train.csv` | 77,299 × 11 | Training data with demand labels |
| `test.csv` | 41,778 × 10 | Test data (no demand column) |
| `sample_submission.csv` | 5 × 2 | Submission format reference |

### Column Descriptions

| Column | Description |
|--------|-------------|
| `Index` | Unique identifier for each datapoint |
| `geohash` | Geographic encoding of a location |
| `day` | Day when the record was captured |
| `timestamp` | Time of the record (format: `H:MM`) |
| `RoadType` | Type of road — Residential / Street / Highway |
| `NumberofLanes` | Number of lanes at the location |
| `LargeVehicles` | Whether large vehicles are permitted |
| `Landmarks` | Whether landmarks are nearby |
| `Temperature` | Temperature at the location |
| `Weather` | Weather condition — Sunny / Rainy / Foggy / Snowy |
| `demand` | **Target** — traffic demand value (0.0 to 1.0) |

---

## Solution Overview

### 1. Preprocessing

- **Timestamp parsing** — Split `H:MM` strings into `hour`, `minute`, and `time_slot` (0–95, one slot per 15 minutes across 24 hours)
- **Missing value imputation:**
  - `Temperature` → filled with training set median
  - `RoadType` → ordinal encoded, unknowns set to `-1`
  - `Weather` → ordinal encoded, unknowns set to `-1`

---

### 2. Feature Engineering

#### a) Cyclical Time Encoding
Raw time slots are linear but time is cyclical — slot 95 (23:45) is adjacent to slot 0 (0:00). We use sine/cosine transforms to capture this:

```python
sin_time = np.sin(2 * np.pi * time_slot / 96)
cos_time = np.cos(2 * np.pi * time_slot / 96)
```

#### b) Geohash Hierarchy
Geohash strings encode geographic coordinates at varying precision. We extract three prefix levels to capture demand patterns at multiple spatial scales:

```python
geo_3 = geohash[:3]   # broad region
geo_4 = geohash[:4]   # sub-region
geo_5 = geohash[:5]   # local area
```

All geohash columns are label-encoded using a **combined train+test vocabulary** to avoid unseen-label errors at inference time.

#### c) Target Encoding (Key Signal)
Mean demand aggregated at multiple levels — computed from training data only:

| Feature | Description |
|---------|-------------|
| `geo_demand_mean` | Mean demand per geohash |
| `geo4_demand_mean` | Mean demand per 4-char geohash prefix |
| `geo5_demand_mean` | Mean demand per 5-char geohash prefix |
| `ts_demand_mean` | Mean demand per time slot (all locations) |
| `geo_ts_mean` | Mean demand per **(geohash × time_slot)** pair — most predictive feature |

> `geo_ts_mean` captures recurring patterns like *"this highway junction always spikes at 8:00 AM"* — the single most informative feature in the model.

#### d) Categorical Encodings

| Column | Encoding |
|--------|----------|
| `LargeVehicles` | Binary: `Allowed → 1`, else `0` |
| `Landmarks` | Binary: `Yes → 1`, else `0` |
| `Weather` | Ordinal: Sunny=0, Rainy=1, Foggy=2, Snowy=3 |
| `RoadType` | Ordinal: Residential=0, Street=1, Highway=2 |

---

### 3. Model — LightGBM

**LightGBM** (Light Gradient Boosting Machine) was chosen for its speed, efficiency, and strong performance on tabular data with mixed feature types.

#### Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `n_estimators` | 1000 | More trees for better convergence |
| `max_depth` | 7 | Enough depth to capture feature interactions |
| `learning_rate` | 0.03 | Low LR paired with high estimator count for stability |
| `num_leaves` | 63 | Controls model complexity (`2^max_depth - 1`) |
| `subsample` | 0.8 | Row sampling to reduce overfitting |
| `colsample_bytree` | 0.8 | Feature sampling per tree |
| `min_child_samples` | 10 | Minimum samples per leaf node |

---

### 4. Final Feature Set (20 features)

```
geo_3_enc, geo_4_enc, geo_5_enc, geohash_enc,
day, sin_time, cos_time, time_slot, hour,
RoadType_enc, NumberofLanes, LargeVehicles_enc,
Landmarks_enc, Temperature, Weather_enc,
geo_demand_mean, ts_demand_mean, geo4_demand_mean,
geo5_demand_mean, geo_ts_mean
```

---

## Results

| Metric | Value |
|--------|-------|
| Train R² | 0.9974 |
| **Hackathon Score** | **99.74 / 100** |

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, manipulation, and aggregation |
| `numpy` | Numerical operations, cyclical encoding, array clipping |
| `scikit-learn` | `LabelEncoder` for geohash encoding, `r2_score` for evaluation |
| `lightgbm` | Gradient boosting regression model |

## Key Insights

- **Location × time is the dominant signal.** The `geo_ts_mean` interaction feature captures highly predictable recurring demand patterns at each geolocation across the day.
- **Geohash hierarchy aids generalization.** Using 3/4/5-char prefixes ensures geohashes with sparse training data still inherit demand signals from their broader region.
- **Cyclical time encoding is essential.** Without sin/cos encoding, the model treats time linearly — breaking the continuity between 23:45 and 00:00.

---
