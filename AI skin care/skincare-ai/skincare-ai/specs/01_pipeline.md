# 01 — Pipeline

## Overview

Seven stages. Stages 01 runs on-device. Stages 02–06 run on the inference server (Triton). Stage 07 is an external LLM API call with a RAG retrieval step.

Total server-side latency target: **under 3 seconds** for a full scan result.

---

## Stage 01 — Face Detection & Alignment

**Runs:** On-device (mobile)  
**Model:** MediaPipe Face Mesh  
**Latency target:** <50ms

### What it does
Detects the face in the camera frame, extracts 468 3D facial landmarks, computes a normalisation transform (rotation, scale, crop), and produces a standardised face image to send to the server.

### Why on-device
Raw camera frames contain full-body and background context — unnecessary data. Normalising on-device means you only transmit the cropped, aligned face crop. This reduces payload size and avoids sending biometric context unnecessarily.

### Input
```
Raw camera frame (RGB, any resolution, typically 1080p or 4K from modern phones)
```

### Output
```json
{
  "face_crop": "<base64-encoded JPEG, 512x512>",
  "landmarks_68": [[x, y, z], ...],   // 68-point subset for pose metadata
  "head_pose": { "yaw": 3.2, "pitch": -1.1, "roll": 0.8 },
  "quality_score": 0.91               // reject if < 0.6
}
```

### Quality gating
Before sending to server, check:
- `quality_score >= 0.6` (MediaPipe confidence)
- `abs(yaw) < 25` and `abs(pitch) < 20` (near-frontal face)
- Minimum face bounding box: 30% of frame width
- Blur detection: Laplacian variance > 100

If any check fails, prompt user to retake ("Move closer / better lighting / face the camera").

### Implementation
```python
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6
)

def process_frame(image_rgb):
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0]
    # align + crop using landmark bounding box
    return aligned_crop, landmarks
```

---

## Stage 02 — Face Region Parsing

**Runs:** Server (GPU)  
**Model:** BiSeNetV2 (pretrained on CelebAMask-HQ, fine-tuned)  
**Latency target:** <80ms  
**Input size:** 512×512

### What it does
Produces a pixel-level semantic segmentation mask dividing the face into named regions. This is what enables zone-level concern mapping ("acne on the chin", "pigmentation on the left cheek").

### Regions (19 classes from CelebAMask-HQ + 4 custom zones)
```
0  background
1  skin (general)
2  left eyebrow
3  right eyebrow
4  left eye
5  right eye
6  nose
7  upper lip
8  inner mouth
9  lower lip
10 left ear
11 right ear
12 hair
13 neck
14 cloth

# Custom zones (fine-tuned):
15 forehead
16 left cheek
17 right cheek
18 chin
19 t-zone (forehead + nose bridge)
```

### Output
```json
{
  "zone_mask": "<base64-encoded PNG, 512x512, single channel>",
  "zone_coverage": {
    "forehead": 0.12,
    "left_cheek": 0.14,
    "right_cheek": 0.13,
    "nose": 0.06,
    "chin": 0.07,
    "t_zone": 0.18,
    "eye_area": 0.08
  }
}
```

The `zone_mask` pixel values correspond to region IDs above. `zone_coverage` is the fraction of total face area each zone occupies.

---

## Stage 03 — Multi-label Concern Detection

**Runs:** Server (GPU)  
**Model:** EfficientNetV2-S (fine-tuned, multi-label)  
**Latency target:** <80ms  
**Input size:** 384×384

### What it detects

| Concern ID | Label | Clinical basis |
|---|---|---|
| 0 | `acne` | Comedones, papules, pustules, cysts |
| 1 | `hyperpigmentation` | Post-inflammatory, melasma, sun spots |
| 2 | `dryness` | Flaking, tight texture, rough surface |
| 3 | `oiliness` | Sebaceous overproduction, shine |
| 4 | `redness` | Rosacea, erythema, sensitivity |
| 5 | `dark_circles` | Periorbital hyperpigmentation or hollowing |
| 6 | `fine_lines` | Surface wrinkles, expression lines |
| 7 | `large_pores` | Dilated follicular openings |
| 8 | `uneven_tone` | Patchy pigmentation without discrete spots |

### Output
```json
{
  "concern_scores": {
    "acne": 0.87,
    "hyperpigmentation": 0.43,
    "dryness": 0.61,
    "oiliness": 0.22,
    "redness": 0.15,
    "dark_circles": 0.71,
    "fine_lines": 0.09,
    "large_pores": 0.38,
    "uneven_tone": 0.29
  },
  "concern_flags": ["acne", "dryness", "dark_circles"],  // score > 0.5 threshold
  "model_version": "efficientnetv2s_v1.2"
}
```

### Thresholding
Default detection threshold: **0.5**. This is tunable per concern — redness and dryness benefit from lower thresholds (0.4) to avoid false negatives. Acne benefits from a higher threshold (0.55) to reduce false positives in early model versions. Document all threshold decisions in the evaluation log.

---

## Stage 04 — Spatial Concern Localisation

**Runs:** Server (GPU)  
**Model:** YOLOv8-seg (fine-tuned) or SAM 2 (prompted, fallback)  
**Latency target:** <150ms  
**Input:** 384×384 face crop + concern_flags from Stage 03

### What it does
Produces a spatial mask per detected concern — where on the face is the acne, where is the pigmentation. This feeds the visual skin map the user sees.

### Two-strategy approach
**Strategy A (preferred once masks are annotated):** YOLOv8-seg fine-tuned on your annotated concern masks. Deterministic, fast, highly accurate.

**Strategy B (use at launch before annotation is complete):** SAM 2 with Stage 03 concern scores as prompt boxes. Zero-shot, slower, but avoids blocking launch on annotation.

Start with Strategy B at trial launch. Switch to Strategy A after the first annotation batch (500+ images) is complete.

### Output
```json
{
  "concern_masks": {
    "acne": "<base64 PNG, 512x512, binary mask>",
    "dryness": "<base64 PNG, 512x512, binary mask>",
    "dark_circles": "<base64 PNG, 512x512, binary mask>"
  },
  "concern_zone_map": {
    "acne": ["chin", "t_zone"],
    "dryness": ["left_cheek", "right_cheek", "forehead"],
    "dark_circles": ["eye_area"]
  },
  "strategy_used": "yolov8seg_v1"
}
```

The `concern_zone_map` is computed by intersecting concern masks with the Stage 02 zone mask.

---

## Stage 05 — Severity Scoring

**Runs:** Server (GPU) — piggybacks Stage 03 forward pass  
**Model:** Regression MLP head on EfficientNetV2-S feature vector  
**Latency:** Near zero (shared forward pass)

### Severity scale
```
0–20    Minimal / trace
21–40   Mild
41–60   Moderate
61–80   Significant
81–100  Severe
```

Acne uses IGA (Investigator's Global Assessment) 0–4 internally, mapped to 0–100 for user-facing output.

### Output
```json
{
  "severity_scores": {
    "acne": { "raw": 2.3, "normalised": 57, "label": "Moderate" },
    "dryness": { "raw": 1.8, "normalised": 44, "label": "Moderate" },
    "dark_circles": { "raw": 2.9, "normalised": 72, "label": "Significant" }
  },
  "overall_skin_score": 68,
  "scan_id": "scan_20240318_u1042",
  "scan_timestamp": "2024-03-18T09:22:31Z"
}
```

`overall_skin_score` is 100 minus a weighted average of severity scores. Higher = better skin.

---

## Stage 06 — Concern → Ingredient Mapping

**Runs:** Application server (no GPU needed)  
**Type:** Deterministic rule engine (not a trained model)  
**Latency:** <10ms

### What it does
Maps detected concerns + severity levels to a prioritised list of active ingredients, filtered by questionnaire data (allergies, skin sensitivity, existing products).

### Mapping table (core)

| Concern | Primary ingredients | Secondary (boosters) | Avoid if |
|---|---|---|---|
| acne | niacinamide, salicylic acid, benzoyl peroxide | azelaic acid, zinc | sensitive skin (BPO) |
| hyperpigmentation | vitamin C, alpha arbutin, niacinamide | kojic acid, tranexamic acid | — |
| dryness | hyaluronic acid, ceramides, glycerin | squalane, shea butter | — |
| oiliness | niacinamide, salicylic acid | clay, green tea | over-stripping actives |
| redness | centella asiatica, azelaic acid | green tea, allantoin | retinol (initially) |
| dark_circles | caffeine, vitamin K, niacinamide | peptides, retinol | — |
| fine_lines | retinol, peptides, hyaluronic acid | vitamin C | pregnancy, rosacea |
| large_pores | niacinamide, salicylic acid, retinol | AHA (glycolic) | — |
| uneven_tone | vitamin C, niacinamide, AHA | retinol, alpha arbutin | — |

This table is authored and validated by the dermatologist partner. It lives in a versioned YAML file, not hardcoded. Dermatologist can update it without a code deploy.

### Output → see [06_schemas.md](./06_schemas.md) `IngredientListResponse`

---

## Stage 07 — LLM Routine Generation

**Runs:** External API (Claude claude-sonnet-4-20250514 or GPT-4o) + internal RAG retrieval  
**Latency target:** <3s (streaming preferred)

### What it does
Takes the `IngredientListResponse` + user's selected products + dermatologist routine templates and produces a personalised morning/night routine in plain language. Also generates the user-facing skin report, ingredient explanations, and progress narratives.

### See [05_llm_rag.md](./05_llm_rag.md) for full implementation.

---

## End-to-end latency budget

| Stage | Where | Target |
|---|---|---|
| 01 Face detection | On-device | <50ms |
| Network upload (512×512 JPEG) | — | <500ms (4G) |
| 02 Region parsing | GPU | <80ms |
| 03 Concern detection | GPU (shared) | <80ms |
| 04 Localisation | GPU | <150ms |
| 05 Severity scoring | GPU (shared) | ~0ms |
| 06 Ingredient mapping | App server | <10ms |
| 07 LLM generation | External API | <3s |
| Network download | — | <200ms |
| **Total (excl. upload)** | | **<3.5s** |

Stages 02–05 run as a single Triton ensemble pipeline (see [07_infrastructure.md](./07_infrastructure.md)). They do not execute sequentially from the client's perspective.
