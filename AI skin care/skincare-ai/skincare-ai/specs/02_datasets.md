# 02 — Datasets

All datasets used across Stages 01–07. Includes download instructions, structure, example records, preprocessing steps, and how each dataset is used in training.

---

## Dataset map

| Dataset | Stage | Task | Size | License |
|---|---|---|---|---|
| CelebAMask-HQ | 02 | Face region segmentation | 30,000 images | CC BY-NC 4.0 |
| ACNE04 | 03, 05 | Acne detection + severity | 1,457 images | Research |
| SCIN (Google) | 03, 05 | Multi-concern, phone photos | 5,000+ images | CC BY 4.0 |
| Fitzpatrick17k | 03 (eval) | Bias testing, skin tone coverage | 16,577 images | CC BY 4.0 |
| SD-198 | 03 (pretrain) | Broad skin condition pretraining | 6,584 images | Research |
| SkinCon | 04 | Concern localisation + explainability | 3,230 images | Research |
| ISIC Archive | 04 (pretrain) | Segmentation mask pretraining | 50,000+ images | CC0 / CC BY |
| FFHQ | 01, 02, 03 | Face diversity, pretraining | 70,000 images | CC BY-NC 2.0 |
| Open Beauty Facts | 06, 07 | Ingredient-to-product mapping | ~200,000 products | ODbL |
| Internal (trial) | 03–07 | In-distribution user scans | TBD | Proprietary |

---

## 1. CelebAMask-HQ

**Used for:** Stage 02 (BiSeNetV2 pretraining — face region parsing)

### Download
```bash
# Google Drive (official)
# https://github.com/switchablenorms/CelebAMask-HQ

pip install gdown
gdown --folder https://drive.google.com/drive/folders/1bqqJiqdzOQ3Vfr5ej_UfxDmFbDyLZHy7

# Structure after download:
CelebAMask-HQ/
├── CelebA-HQ-img/          # 30,000 high-res face images (1024x1024)
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── CelebAMask-HQ-mask-anno/
│   ├── 0/                  # Annotations for images 0-2999
│   │   ├── 00000_skin.png
│   │   ├── 00000_l_brow.png
│   │   ├── 00000_r_brow.png
│   │   └── ...             # One PNG per class per image
│   └── ...
└── CelebA-HQ-to-CelebA-mapping.txt
```

### Mask classes (raw filenames → our region IDs)
```python
CELEBA_MASK_CLASSES = {
    'skin': 1,
    'l_brow': 2, 'r_brow': 3,
    'l_eye': 4, 'r_eye': 5,
    'nose': 6,
    'u_lip': 7, 'mouth': 8, 'l_lip': 9,
    'l_ear': 10, 'r_ear': 11,
    'hair': 12,
    'neck': 13,
    'cloth': 14
}
# Zones 15-19 (forehead, cheeks, chin, t-zone) are added via fine-tuning
# on internally annotated data — see 04_annotation.md
```

### Preprocessing script
```python
import cv2
import numpy as np
from pathlib import Path

def merge_celeba_masks(image_id: str, mask_dir: Path) -> np.ndarray:
    """Merge per-class PNGs into a single H×W mask."""
    merged = np.zeros((512, 512), dtype=np.uint8)
    for class_name, class_id in CELEBA_MASK_CLASSES.items():
        mask_path = mask_dir / f"{image_id}_{class_name}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (512, 512))
            merged[mask > 0] = class_id
    return merged

# Resize images from 1024→512
def preprocess_image(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
```

### Train/val/test split
```
Train: images 0–26,999   (90%)
Val:   images 27,000–28,999  (6.7%)
Test:  images 29,000–29,999  (3.3%)
```

---

## 2. ACNE04

**Used for:** Stage 03 (acne detection), Stage 05 (severity scoring)

### Download
```bash
# Paper: https://arxiv.org/abs/2008.00527
# Dataset: https://github.com/xpwu95/LDL
# Request access via the GitHub repo or email the authors directly.

# Structure:
ACNE04/
├── Classification/
│   ├── Grade_1/            # Mild (comedonal) — 459 images
│   ├── Grade_2/            # Moderate (papular/pustular) — 289 images
│   ├── Grade_3/            # Severe (nodulocystic) — 323 images
│   └── Grade_4/            # Very severe — 386 images
└── Detection/
    ├── images/             # Face images (varying resolution)
    └── labels/             # PASCAL VOC XML annotations (lesion bounding boxes)
```

### Example record (Detection/labels/0001.xml)
```xml
<annotation>
  <filename>0001.jpg</filename>
  <size><width>640</width><height>480</height></size>
  <object>
    <name>acne</name>
    <bndbox><xmin>123</xmin><ymin>89</ymin><xmax>145</xmax><ymax>112</ymax></bndbox>
  </object>
  <object>
    <name>acne</name>
    <bndbox><xmin>201</xmin><ymin>145</ymin><xmax>220</xmax><ymax>167</ymax></bndbox>
  </object>
</annotation>
```

### IGA → severity mapping
```python
IGA_TO_SEVERITY = {
    0: 0,    # Clear
    1: 20,   # Almost clear
    2: 45,   # Mild (Grade_1)
    3: 65,   # Moderate (Grade_2)
    4: 100   # Severe (Grade_3 / Grade_4)
}
```

### Preprocessing
```python
import xml.etree.ElementTree as ET

def load_acne04_detection(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        boxes.append({
            'label': 'acne',
            'xmin': int(bbox.find('xmin').text),
            'ymin': int(bbox.find('ymin').text),
            'xmax': int(bbox.find('xmax').text),
            'ymax': int(bbox.find('ymax').text),
        })
    return boxes
```

### Note on dataset size
1,457 images is too small for a robust model on its own. Use it in combination with SCIN. Apply heavy augmentation (see augmentation section below).

---

## 3. SCIN (Skin Condition Image Network — Google)

**Used for:** Stage 03 (multi-label concern detection, primary training set)

This is your most important dataset. Phone-camera photos, diverse skin tones, consumer context — closest to your actual user input.

### Download
```bash
git clone https://github.com/google-research-datasets/scin
cd scin

# Dataset requires accepting terms at:
# https://datasetsearch.research.google.com/search?query=scin+skin
# Download link provided after terms acceptance.

# Structure:
scin/
├── images/
│   ├── train/
│   └── test/
├── scin_train.jsonl         # Training metadata
├── scin_test.jsonl          # Test metadata
└── label_definitions.json  # Condition taxonomy
```

### Example record (scin_train.jsonl)
```json
{
  "image_id": "scin_00012847",
  "image_path": "images/train/scin_00012847.jpg",
  "conditions": ["acne", "post_inflammatory_hyperpigmentation"],
  "fitzpatrick_scale": 4,
  "body_part": "face",
  "source": "consumer_upload",
  "dermatologist_verified": true,
  "metadata": {
    "age_range": "18-30",
    "gender": "female"
  }
}
```

### Mapping SCIN conditions → your concern labels
```python
SCIN_TO_CONCERN = {
    "acne": "acne",
    "comedones": "acne",
    "cystic_acne": "acne",
    "post_inflammatory_hyperpigmentation": "hyperpigmentation",
    "melasma": "hyperpigmentation",
    "solar_lentigo": "hyperpigmentation",
    "xerosis": "dryness",
    "seborrhea": "oiliness",
    "rosacea": "redness",
    "erythema": "redness",
    "periorbital_hyperpigmentation": "dark_circles",
    "rhytides": "fine_lines",
    "enlarged_pores": "large_pores",
    "dyschromia": "uneven_tone",
}
```

---

## 4. Fitzpatrick17k

**Used for:** Bias evaluation and fine-tuning for skin tone coverage

### Download
```bash
git clone https://github.com/mattgroh/fitzpatrick17k
# Images link to dermatology websites — use the download script provided

# Structure:
fitzpatrick17k/
├── data/
│   └── fitzpatrick17k.csv   # Metadata with labels
└── download_images.py
```

### Example record (fitzpatrick17k.csv)
```
image_id,url,label,fitzpatrick_scale,nine_partition_label,three_partition_label
abc123,https://dermnet.com/...,acne,3,inflammatory,benign
def456,https://dermnet.com/...,melasma,5,pigmented,benign
```

### Fitzpatrick scale reference
```
FST I   — Very fair, always burns, never tans
FST II  — Fair, usually burns, sometimes tans
FST III — Medium, sometimes burns, always tans
FST IV  — Olive, rarely burns, always tans
FST V   — Brown, very rarely burns
FST VI  — Dark brown/black, never burns
```

**Your models must be evaluated separately on each FST group.** See [08_evaluation.md](./08_evaluation.md).

---

## 5. Open Beauty Facts

**Used for:** Stage 06 (ingredient-to-product database), Stage 07 (RAG retrieval)

### Download
```bash
# Full database dump (CSV):
wget https://world.openbeautyfacts.org/data/en.openbeautyfacts.org.products.csv.gz
gunzip en.openbeautyfacts.org.products.csv.gz

# Or use the API:
curl "https://world.openbeautyfacts.org/api/v0/product/737628064502.json"
```

### Example record (CSV columns of interest)
```
product_name: "CeraVe Moisturising Cream"
brands: "CeraVe"
categories: "Moisturiser, Face cream"
ingredients_text: "Aqua, Glycerin, Cetearyl Alcohol, Caprylic/Capric Triglyceride, Ceramide NP, Ceramide AP, Ceramide EOP, Cholesterol, Niacinamide, ..."
```

### Preprocessing pipeline
```python
import pandas as pd

def load_beauty_facts(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep='\t', on_bad_lines='skip',
                     usecols=['product_name', 'brands', 'categories',
                              'ingredients_text', 'countries'])
    # Filter to face products
    df = df[df['categories'].str.contains('face|skin|moistur|serum|cleanser',
                                           case=False, na=False)]
    # Drop rows with no ingredient list
    df = df.dropna(subset=['ingredients_text'])
    # Normalise ingredient names
    df['ingredients_list'] = df['ingredients_text'].apply(parse_inci_ingredients)
    return df

def parse_inci_ingredients(raw: str) -> list:
    """Parse INCI ingredient string into clean list."""
    import re
    ingredients = re.split(r',(?![^(]*\))', raw)
    return [i.strip().lower() for i in ingredients if len(i.strip()) > 2]
```

---

## Augmentation strategy

All training images go through the following augmentation pipeline. This is critical because:
- Your users take photos in bathroom lighting (warm, uneven)
- Phone cameras apply variable post-processing
- The public datasets are often clinical-quality (better than real user input)

```python
import albumentations as A

TRAIN_AUGMENTATION = A.Compose([
    # Geometric
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.3),
    A.RandomScale(scale_limit=0.1, p=0.3),

    # Lighting (most important for your use case)
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1, p=0.5),
    A.RandomShadow(p=0.3),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),

    # Blur (simulate motion / focus issues)
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5)),
        A.MotionBlur(blur_limit=5),
    ], p=0.3),

    # Compression (JPEG artefacts from phone processing)
    A.ImageCompression(quality_lower=70, quality_upper=95, p=0.5),

    # Skin tone calibration (simulate colour temperature shifts)
    A.RGBShift(r_shift_limit=15, g_shift_limit=10, b_shift_limit=20, p=0.4),

    # Normalise last
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

## Internal dataset (free trial)

This becomes your most valuable asset. Design the collection pipeline before the trial launches.

### Collection schema per scan
```json
{
  "scan_id": "uuid",
  "user_id": "uuid (anonymised)",
  "timestamp": "ISO8601",
  "face_crop_path": "s3://bucket/scans/scan_id.jpg",
  "fitzpatrick_self_reported": 3,
  "questionnaire": {
    "concerns_self_reported": ["acne", "dryness"],
    "allergies": ["fragrance"],
    "skin_type_self_reported": "combination"
  },
  "annotation_status": "pending | in_review | complete",
  "annotations": {
    "concern_labels": ["acne", "dryness"],
    "severity_labels": { "acne": 2, "dryness": 1 },
    "annotator_id": "derm_001",
    "annotation_timestamp": "ISO8601"
  }
}
```

### Target: 300 annotated scans from free trial → first model fine-tune
