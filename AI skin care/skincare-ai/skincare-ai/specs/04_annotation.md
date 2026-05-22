# 04 — Annotation Guide

This file covers everything needed to produce labelled training data — what to annotate, how to annotate it, the tooling setup, quality control, and the dermatologist review process.

---

## What needs annotation

| Annotation type | Used by | Volume target | Who annotates | Dermatologist review |
|---|---|---|---|---|
| Concern classification labels | Stage 03 | 3,000–5,000 images | Trained annotator | Sample QA (20%) |
| Severity labels (IGA scale) | Stage 05 | 3,000 images | Dermatologist preferred | Full review |
| Zone masks (face regions) | Stage 02 | 500 images | Trained annotator | Sample QA (10%) |
| Spatial concern masks | Stage 04 | 1,000 images | Dermatologist preferred | Full review |
| Routine templates (text) | Stage 07 | 20–30 templates | Dermatologist | Full review |
| Ingredient conflict rules | Stage 06 | Full table | Dermatologist | Full review |

---

## Tooling — Label Studio

Use **Label Studio** for all image annotation. It's open source, self-hostable, and supports all annotation types needed (classification, bounding boxes, polygons, segmentation masks).

### Setup
```bash
pip install label-studio

# Start the server:
label-studio start --port 8080

# Or with Docker:
docker run -it -p 8080:8080 \
  -v $(pwd)/label-studio-data:/label-studio/data \
  heartexlabs/label-studio:latest
```

### Project setup for concern classification
```xml
<!-- Label config for concern classification (paste into Label Studio project settings) -->
<View>
  <Image name="image" value="$image"/>
  <Header value="Skin Concerns (select all that apply)"/>
  <Choices name="concerns" toName="image" choice="multiple">
    <Choice value="acne"/>
    <Choice value="hyperpigmentation"/>
    <Choice value="dryness"/>
    <Choice value="oiliness"/>
    <Choice value="redness"/>
    <Choice value="dark_circles"/>
    <Choice value="fine_lines"/>
    <Choice value="large_pores"/>
    <Choice value="uneven_tone"/>
    <Choice value="none_visible"/>
  </Choices>
  <Header value="Overall Severity"/>
  <Rating name="overall_severity" toName="image" maxRating="5" icon="star"/>
  <Header value="Annotator confidence"/>
  <Choices name="confidence" toName="image" choice="single">
    <Choice value="high"/>
    <Choice value="medium"/>
    <Choice value="low_refer_to_derm"/>
  </Choices>
</View>
```

### Project setup for spatial masks
```xml
<View>
  <Image name="image" value="$image"/>
  <Header value="Draw masks around each concern area"/>
  <BrushLabels name="masks" toName="image">
    <Label value="acne" background="#FF6B6B"/>
    <Label value="hyperpigmentation" background="#4ECDC4"/>
    <Label value="dryness" background="#45B7D1"/>
    <Label value="redness" background="#FFA07A"/>
    <Label value="dark_circles" background="#9B59B6"/>
    <Label value="large_pores" background="#F39C12"/>
  </BrushLabels>
</View>
```

---

## Annotation schema

### Concern classification label (JSON export from Label Studio)
```json
{
  "image_id": "scin_00012847",
  "image_path": "s3://skincare-data/annotations/scin_00012847.jpg",
  "annotator_id": "ann_003",
  "annotation_timestamp": "2024-03-18T11:22:00Z",
  "concerns": ["acne", "hyperpigmentation"],
  "none_visible": false,
  "overall_severity": 3,
  "confidence": "high",
  "notes": "Acne concentrated on chin and forehead. Faint PIH on left cheek."
}
```

### Severity label
```json
{
  "image_id": "scin_00012847",
  "annotator_id": "derm_001",
  "severity_labels": {
    "acne": {
      "iga_score": 2,         // 0=clear, 1=almost clear, 2=mild, 3=moderate, 4=severe
      "normalised_score": 45, // Maps to 0-100 scale
      "lesion_count_estimate": "10-20",
      "lesion_types": ["papules", "comedones"]
    },
    "hyperpigmentation": {
      "iga_score": 1,
      "normalised_score": 20,
      "coverage_estimate": "5-15%",
      "pattern": "post_inflammatory"
    }
  }
}
```

### Spatial mask export
```json
{
  "image_id": "scin_00012847",
  "image_size": [512, 512],
  "annotator_id": "derm_001",
  "masks": [
    {
      "concern": "acne",
      "format": "RLE",             // Run-length encoding
      "data": "...",               // Label Studio RLE output
      "polygon_approx": [[x1,y1], [x2,y2], ...]  // Simplified polygon for YOLO
    }
  ]
}
```

---

## Annotation guidelines (hand to annotators)

### General rules
1. Annotate what is **visible in the photo**, not what the person reports in the questionnaire.
2. When in doubt, annotate conservatively — do not add a label unless the concern is clearly visible.
3. If image quality is too poor (blurry, dark, extreme angle), mark as `reject` and do not annotate.
4. Mark your confidence. Low-confidence annotations go to dermatologist review before entering training.

### Concern-specific guidance

**Acne**
- Include: visible papules (red bumps), pustules (whitehead), comedones (blackheads), cysts
- Exclude: general redness without lesions, scars from old acne
- Common mistake: confusing large pores with comedones — pores are open, flat; comedones have a plug

**Hyperpigmentation**
- Include: dark spots darker than surrounding skin, melasma patches, PIH (brown marks left after acne)
- Exclude: natural skin variation, freckles unless clearly uneven
- Common mistake: confusing hyperpigmentation with redness — pigmentation = brown/dark, redness = red/pink

**Dryness**
- Include: visible flaking, rough texture visible in the photo, tight/creased appearance
- Exclude: normal dry-looking skin without visible texture change — do not guess
- Note: very hard to judge in JPEG photos. Only label if clearly visible.

**Redness**
- Include: diffuse red/pink tone, visible broken capillaries (small red lines), rosacea-pattern flushing
- Exclude: natural complexion warmth in FST I–II (common annotator error)
- Critical: FST V–VI do not show redness the same way. Redness in darker skin tones appears as deeper discolouration, not pink/red. If unsure, mark `low_refer_to_derm`.

**Dark circles**
- Include: visible under-eye darkness darker than cheek area, bluish or brownish discolouration
- Exclude: under-eye hollowing/shadows without pigmentation change (this is structural, not a pigment concern)

**Large pores**
- Include: clearly visible open pores on nose, cheeks. Annotate only if texture is clearly visible.
- Exclude: general skin texture, follicles

---

## Quality control process

### Tier 1: Annotator self-check
Before submitting, annotator confirms:
- Image is usable quality (not rejected)
- All visible concerns are labelled
- Confidence is accurate

### Tier 2: Inter-annotator agreement
For the first 200 images, have two annotators label independently. Compute agreement:
```python
from sklearn.metrics import jaccard_score
import numpy as np

def compute_iaa(labels_a: list, labels_b: list, concern_labels: list) -> float:
    """Cohen's kappa for multi-label annotation agreement."""
    from sklearn.metrics import cohen_kappa_score
    # Flatten to per-concern binary vectors
    vec_a = [[1 if c in l else 0 for c in concern_labels] for l in labels_a]
    vec_b = [[1 if c in l else 0 for c in concern_labels] for l in labels_b]
    kappas = [cohen_kappa_score(np.array(vec_a)[:, i], np.array(vec_b)[:, i])
              for i in range(len(concern_labels))]
    return np.mean(kappas)
```
**Target:** Kappa > 0.65. Below that, re-train the annotator or revise the guidelines.

### Tier 3: Dermatologist QA
- All severity labels: 100% review
- All spatial masks: 100% review
- Classification labels with `confidence = low_refer_to_derm`: 100% review
- Random sample of `high` confidence labels: 20% review

### Tier 4: Automated checks
```python
def validate_annotation(annotation: dict) -> list:
    """Return list of issues. Empty list = pass."""
    issues = []
    if annotation.get('none_visible') and annotation.get('concerns'):
        issues.append("none_visible=True but concerns are labelled")
    if not annotation.get('confidence'):
        issues.append("Missing confidence rating")
    if annotation.get('concerns') and not annotation.get('overall_severity'):
        issues.append("Concerns labelled but no severity rating")
    return issues
```

---

## Dermatologist knowledge base (for Stage 06 + 07)

### What the dermatologist needs to produce

**1. Concern → routine mapping table**
A YAML file mapping concern + severity level to a morning and night routine template. The ML engineer provides the schema; the dermatologist fills in the clinical content.

```yaml
# data/clinical/concern_routines.yaml
acne:
  mild:
    morning:
      steps:
        - step: cleanser
          instruction: "Gently cleanse with a salicylic acid cleanser (1-2%). Massage for 60 seconds, rinse with lukewarm water."
          key_ingredient: salicylic_acid
          frequency: daily
        - step: moisturiser
          instruction: "Apply a lightweight, non-comedogenic moisturiser. Look for niacinamide (4-5%) to help regulate sebum."
          key_ingredient: niacinamide
          frequency: daily
        - step: spf
          instruction: "Apply SPF 30+ mineral sunscreen. Essential even when using actives."
          key_ingredient: zinc_oxide
          frequency: daily
    night:
      steps:
        - step: cleanser
          instruction: "Double cleanse if wearing SPF or makeup. Oil cleanser first, then gentle foaming cleanser."
          key_ingredient: null
          frequency: daily
        - step: treatment
          instruction: "Apply benzoyl peroxide spot treatment (2.5%) only to active lesions. Start 3x/week, build up."
          key_ingredient: benzoyl_peroxide
          frequency: "3x/week initially"
          caution: "Do not apply all over. Avoid if pregnant. Can bleach fabrics."
```

**2. Ingredient conflict table**

```yaml
# data/clinical/ingredient_conflicts.yaml
conflicts:
  - ingredients: [retinol, benzoyl_peroxide]
    reason: "Benzoyl peroxide can oxidise and deactivate retinol."
    resolution: "Use retinol AM, BPO PM. Or alternate nights."

  - ingredients: [vitamin_c, niacinamide]
    reason: "High concentrations can form niacin, causing flushing. Low-dose co-use is generally fine."
    resolution: "Use vitamin C AM, niacinamide PM. Or use formulations ≤5% niacinamide."
    severity: mild   # Not a hard conflict

  - ingredients: [aha, retinol]
    reason: "Both exfoliate. Combined use causes irritation and barrier damage."
    resolution: "Alternate nights. Never layer same evening."

  - ingredients: [vitamin_c, aha]
    reason: "Both low pH. Can over-acidify skin if layered."
    resolution: "Separate by 20-30 minutes, or use on alternate days."
```
