# 08 — Evaluation

How we know the models are good enough before they touch users. Every model must pass its evaluation criteria before being registered for production.

---

## Stage 02 — BiSeNetV2 (Face Parsing)

### Metrics
- **mIoU** (mean Intersection over Union) — primary metric
- **Per-class IoU** — track each zone separately

### Evaluation dataset
```
Test split of CelebAMask-HQ (images 29,000–29,999)
+ 100 internally annotated images with custom zones
```

### Pass threshold
```
mIoU (14 original classes): ≥ 0.70
mIoU (custom zones — forehead, cheeks, chin): ≥ 0.60
```

### Evaluation script
```python
import numpy as np
from sklearn.metrics import jaccard_score

def compute_miou(pred_masks: np.ndarray, true_masks: np.ndarray,
                 n_classes: int = 19) -> dict:
    ious = []
    per_class = {}
    for cls in range(n_classes):
        pred_cls = (pred_masks == cls).flatten()
        true_cls = (true_masks == cls).flatten()
        if true_cls.sum() == 0:
            continue  # Skip absent classes
        iou = jaccard_score(true_cls, pred_cls, zero_division=0)
        ious.append(iou)
        per_class[cls] = round(iou, 3)

    return {"mean_iou": round(np.mean(ious), 3), "per_class": per_class}
```

---

## Stage 03 — EfficientNetV2-S (Concern Detection)

### Metrics
- **AUC-ROC per concern** — primary
- **F1 @ threshold=0.5** per concern
- **Precision / Recall** per concern
- **Per-Fitzpatrick-group AUC** — mandatory bias check

### Evaluation datasets
```
Primary test set: SCIN test split (held out, never seen during training)
Bias test set:    Fitzpatrick17k (entire dataset — evaluate, never train on)
```

### Pass thresholds
```
Mean AUC across 9 concerns:     ≥ 0.78
Acne AUC:                       ≥ 0.83
Worst single-concern AUC:       ≥ 0.68
FST I–II AUC:                   ≥ 0.78
FST III–IV AUC:                 ≥ 0.76
FST V–VI AUC:                   ≥ 0.72    ← Hard blocker. Do not ship below this.
Max AUC gap (best vs worst FST): ≤ 0.12
```

### Evaluation script
```python
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import pandas as pd

CONCERN_LABELS = [
    'acne', 'hyperpigmentation', 'dryness', 'oiliness',
    'redness', 'dark_circles', 'fine_lines', 'large_pores', 'uneven_tone'
]

def evaluate_concern_model(model, dataloader, device):
    model.eval()
    all_probs, all_labels, all_fitzpatrick = [], [], []

    with torch.no_grad():
        for images, labels, fitzpatrick in dataloader:
            logits, _ = model(images.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            all_fitzpatrick.extend(fitzpatrick.numpy())

    probs = np.vstack(all_probs)
    labels = np.vstack(all_labels)
    fitzpatrick = np.array(all_fitzpatrick)

    results = {}

    # Per-concern AUC
    for i, concern in enumerate(CONCERN_LABELS):
        if labels[:, i].sum() == 0:
            continue
        auc = roc_auc_score(labels[:, i], probs[:, i])
        f1 = f1_score(labels[:, i], probs[:, i] > 0.5, zero_division=0)
        results[concern] = {
            "auc": round(auc, 3),
            "f1": round(f1, 3),
            "precision": round(precision_score(labels[:, i], probs[:, i] > 0.5, zero_division=0), 3),
            "recall": round(recall_score(labels[:, i], probs[:, i] > 0.5, zero_division=0), 3),
            "n_positive": int(labels[:, i].sum())
        }

    results["mean_auc"] = round(np.mean([v["auc"] for v in results.values()
                                          if isinstance(v, dict)]), 3)

    # Per-Fitzpatrick AUC
    fitzpatrick_results = {}
    for fst in range(1, 7):
        mask = fitzpatrick == fst
        if mask.sum() < 10:
            continue
        fst_aucs = []
        for i, concern in enumerate(CONCERN_LABELS):
            if labels[mask, i].sum() == 0:
                continue
            fst_aucs.append(roc_auc_score(labels[mask, i], probs[mask, i]))
        if fst_aucs:
            fitzpatrick_results[f"FST_{fst}"] = round(np.mean(fst_aucs), 3)

    results["fitzpatrick_auc"] = fitzpatrick_results
    results["fitzpatrick_max_gap"] = round(
        max(fitzpatrick_results.values()) - min(fitzpatrick_results.values()), 3
    )

    return results
```

### Example evaluation report
```json
{
  "model_version": "efficientnetv2s_v1.2",
  "eval_dataset": "scin_test + fitzpatrick17k",
  "mean_auc": 0.812,
  "per_concern": {
    "acne":             { "auc": 0.871, "f1": 0.794, "n_positive": 412 },
    "hyperpigmentation":{ "auc": 0.823, "f1": 0.741, "n_positive": 298 },
    "dryness":          { "auc": 0.798, "f1": 0.712, "n_positive": 334 },
    "redness":          { "auc": 0.743, "f1": 0.668, "n_positive": 156 },
    "dark_circles":     { "auc": 0.834, "f1": 0.771, "n_positive": 289 }
  },
  "fitzpatrick_auc": {
    "FST_1": 0.841,
    "FST_2": 0.829,
    "FST_3": 0.812,
    "FST_4": 0.798,
    "FST_5": 0.774,
    "FST_6": 0.751
  },
  "fitzpatrick_max_gap": 0.090,
  "pass": true
}
```

---

## Stage 04 — YOLOv8-seg (Localisation)

### Metrics
- **mAP50** (mean Average Precision @ IoU 0.5) — primary
- **mAP50-95** — secondary
- **Per-concern AP**

### Pass threshold
```
mAP50:   ≥ 0.55
mAP50-95: ≥ 0.35
```

### Evaluation (built into YOLO)
```bash
yolo segment val \
  model=runs/skin_seg/v1/weights/best.pt \
  data=configs/skin_seg_data.yaml \
  split=test \
  save_json=true
```

---

## Stage 05 — Severity Scoring

### Metrics
- **Pearson correlation** vs dermatologist IGA scores
- **MAE** (Mean Absolute Error) in normalised 0–100 scale
- **IGA concordance** (% within 1 IGA grade of dermatologist)

### Pass threshold
```
Acne Pearson correlation vs IGA: ≥ 0.75
Mean MAE across concerns:        ≤ 12 points (0-100 scale)
IGA concordance:                 ≥ 80%
```

```python
from scipy import stats

def evaluate_severity(pred_scores: np.ndarray, true_iga: np.ndarray) -> dict:
    # Convert IGA (0-4) to 0-100
    true_normalised = (true_iga / 4) * 100

    pearson_r, p_value = stats.pearsonr(pred_scores, true_normalised)
    mae = np.mean(np.abs(pred_scores - true_normalised))

    # IGA concordance: pred within ±25 points (= 1 IGA grade) of true
    concordance = np.mean(np.abs(pred_scores - true_normalised) <= 25)

    return {
        "pearson_r": round(pearson_r, 3),
        "p_value": round(p_value, 4),
        "mae": round(mae, 2),
        "iga_concordance": round(concordance, 3)
    }
```

---

## LLM evaluation rubric

50 test cases per template, scored by dermatologist before any template goes live.

### Test cases structure
```json
{
  "test_id": "llm_eval_001",
  "input": {
    "severity_report": { ... },
    "zone_map": { ... }
  },
  "llm_output": { ... },
  "dermatologist_scores": {
    "clinical_accuracy": 4,          // 1-5: Is the information correct?
    "tone": 5,                        // 1-5: Non-alarmist, supportive?
    "actionability": 4,              // 1-5: Does the user know what to do?
    "false_claims": "pass",          // pass/fail: Any overstatements or misinformation?
    "schema_valid": "pass",          // pass/fail: Valid JSON?
    "notes": "Retinol caution was clear and appropriately cautious."
  }
}
```

### Aggregated pass criteria
```python
def evaluate_llm_template(test_cases: list) -> dict:
    scores = {
        "clinical_accuracy": [],
        "tone": [],
        "actionability": [],
        "false_claims_pass_rate": [],
        "schema_valid_rate": []
    }

    for case in test_cases:
        s = case["dermatologist_scores"]
        scores["clinical_accuracy"].append(s["clinical_accuracy"])
        scores["tone"].append(s["tone"])
        scores["actionability"].append(s["actionability"])
        scores["false_claims_pass_rate"].append(1 if s["false_claims"] == "pass" else 0)
        scores["schema_valid_rate"].append(1 if s["schema_valid"] == "pass" else 0)

    result = {k: round(np.mean(v), 2) for k, v in scores.items()}
    result["pass"] = (
        result["clinical_accuracy"] >= 4.0 and
        result["tone"] >= 4.0 and
        result["actionability"] >= 4.0 and
        result["false_claims_pass_rate"] == 1.0 and  # Zero tolerance
        result["schema_valid_rate"] == 1.0
    )
    return result
```

---

## Regression testing

Every new model version must beat the current production version on the primary metric before it can be deployed.

```python
def regression_check(new_metrics: dict, prod_metrics: dict,
                     tolerance: float = 0.01) -> bool:
    """
    New model must match or exceed production on all key metrics.
    tolerance: allow 1% degradation on secondary metrics.
    """
    hard_metrics = ["mean_auc", "fitzpatrick_max_gap"]
    soft_metrics = ["acne_auc", "redness_auc"]

    for metric in hard_metrics:
        if new_metrics[metric] < prod_metrics[metric]:
            print(f"FAIL: {metric} regressed: {new_metrics[metric]} < {prod_metrics[metric]}")
            return False

    for metric in soft_metrics:
        if new_metrics[metric] < prod_metrics[metric] - tolerance:
            print(f"FAIL: {metric} soft regression: {new_metrics[metric]}")
            return False

    return True
```
