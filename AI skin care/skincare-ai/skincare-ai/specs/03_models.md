# 03 — Models

Architecture, weights, training config, and expected metrics for every model in the pipeline.

---

## Stage 02 — BiSeNetV2 (Face Region Parsing)

### Pretrained weights
```bash
# From the official repo:
git clone https://github.com/CoinCheung/BiSeNet
cd BiSeNet

# Download CelebAMask-HQ pretrained weights:
# https://drive.google.com/file/d/1ykNpS5dBqbhOmJx7HBPV4JiIbGidwqRr
wget -O checkpoints/bisenetv2_celebamask.pth <google_drive_link>
```

### Fine-tuning for custom zones

You need to add 5 custom zone classes (forehead, left cheek, right cheek, chin, t-zone) on top of the 14 CelebAMask-HQ classes. This requires annotated masks for those zones (see [04_annotation.md](./04_annotation.md)).

```python
import torch
import torch.nn as nn
from bisenet import BiSeNetV2

class BiSeNetV2Extended(nn.Module):
    def __init__(self, n_classes=19):  # 14 original + 5 custom zones
        super().__init__()
        self.model = BiSeNetV2(n_classes=n_classes)

    def forward(self, x):
        return self.model(x)

def load_pretrained_extended(checkpoint_path, n_classes=19):
    model = BiSeNetV2Extended(n_classes=n_classes)
    checkpoint = torch.load(checkpoint_path)
    # Load everything except the final classification head
    state_dict = {k: v for k, v in checkpoint.items()
                  if 'head' not in k}
    model.load_state_dict(state_dict, strict=False)
    return model
```

### Training config (fine-tune)
```yaml
# configs/bisenetv2_finetune.yaml
model:
  n_classes: 19
  pretrained: checkpoints/bisenetv2_celebamask.pth
  freeze_backbone: false       # Unfreeze after 5 epochs

training:
  batch_size: 16
  epochs: 30
  optimizer: SGD
  lr: 0.001
  lr_schedule: poly           # lr * (1 - iter/max_iter)^0.9
  momentum: 0.9
  weight_decay: 0.0005
  loss: OHEMCrossEntropyLoss  # Online Hard Example Mining
  loss_weights: [1.0, 1.0, 1.0, 1.0, 1.0]  # Auxiliary heads

data:
  train_dir: data/bisenet/train
  val_dir: data/bisenet/val
  input_size: [512, 512]
  augmentation: true

hardware:
  gpus: 1
  mixed_precision: true
```

### Expected metrics
```
mIoU on CelebAMask-HQ test: >70%
mIoU on custom zones (after fine-tune): >60%
Inference latency (A10 GPU): ~45ms @ batch=1
```

---

## Stage 03 — EfficientNetV2-S (Multi-label Concern Detection)

### Pretrained weights
```python
import timm

# Load ImageNet pretrained EfficientNetV2-S
model = timm.create_model(
    'tf_efficientnetv2_s',
    pretrained=True,           # Downloads from HuggingFace hub
    num_classes=0,             # Remove classification head
    global_pool='avg'
)
# model output: [batch, 1280] feature vector
```

### Custom classification head
```python
class SkinConcernModel(nn.Module):
    def __init__(self, backbone, n_concerns=9, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_concerns)
            # No sigmoid here — use BCEWithLogitsLoss during training
            # Apply sigmoid at inference
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features), features  # Return features for Stage 05
```

### Training script
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import mlflow

CONCERN_LABELS = [
    'acne', 'hyperpigmentation', 'dryness', 'oiliness',
    'redness', 'dark_circles', 'fine_lines', 'large_pores', 'uneven_tone'
]

# Weighted loss — give more weight to rare concerns
CONCERN_WEIGHTS = torch.tensor([
    1.0,   # acne (common)
    1.2,   # hyperpigmentation
    1.0,   # dryness
    1.1,   # oiliness
    1.5,   # redness (less annotated)
    1.3,   # dark_circles
    1.8,   # fine_lines (rare in young cohort)
    1.4,   # large_pores
    1.3,   # uneven_tone
])

def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=CONCERN_WEIGHTS.to(device))
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()

        with autocast():
            logits, _ = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(loader)
```

### Training config
```yaml
# configs/efficientnetv2s_concerns.yaml
model:
  backbone: tf_efficientnetv2_s
  pretrained: imagenet
  n_concerns: 9
  dropout: 0.3

training:
  batch_size: 32
  epochs: 50
  optimizer: AdamW
  lr: 0.0001
  weight_decay: 0.01
  lr_schedule: cosine_with_warmup
  warmup_epochs: 3
  loss: BCEWithLogitsLoss
  early_stopping_patience: 8
  mixed_precision: true

data:
  train_datasets: [acne04, scin, sd198]
  val_datasets: [scin_val]
  test_datasets: [fitzpatrick17k]   # Held-out bias test
  input_size: [384, 384]
  augmentation: heavy

experiment:
  tracking: mlflow
  run_name: efficientnetv2s_v1
  save_top_k: 3
  metric: val_mean_auc
```

### Dataset mixing strategy
```python
from torch.utils.data import ConcatDataset, WeightedRandomSampler

# Different datasets have different quality levels — weight accordingly
train_datasets = {
    'acne04': (load_acne04(), 2.0),      # High quality, oversample
    'scin': (load_scin_train(), 1.5),    # Highest priority
    'sd198': (load_sd198(), 0.5),        # Lower quality, undersample
    'internal': (load_internal(), 3.0),  # When available — highest priority
}

def build_weighted_sampler(datasets_with_weights):
    all_samples = []
    for dataset, weight in datasets_with_weights.values():
        all_samples.extend([weight] * len(dataset))
    return WeightedRandomSampler(all_samples, len(all_samples))
```

### Expected metrics (after fine-tune on public data)
```
Mean AUC across 9 concerns:   >0.78
Acne AUC:                     >0.85 (most data)
Redness AUC:                  >0.72 (least data)
Fine lines AUC:               >0.70

Per Fitzpatrick group (target):
  FST I–II:    AUC > 0.80
  FST III–IV:  AUC > 0.78
  FST V–VI:    AUC > 0.74  ← watch this closely
```

---

## Stage 04 — YOLOv8-seg (Spatial Localisation)

### Setup
```bash
pip install ultralytics

# Download YOLOv8 segmentation pretrained weights:
from ultralytics import YOLO
model = YOLO('yolov8m-seg.pt')  # Medium size — good accuracy/speed tradeoff
```

### Dataset format (YOLO segmentation)
```
# Directory structure:
data/yolo_seg/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/          # .txt files with polygon masks
    └── val/

# Label format (per line in .txt):
# class_id x1 y1 x2 y2 x3 y3 ... xn yn
# All coordinates normalised 0–1, polygon points clockwise

# Example (acne lesion mask):
0 0.312 0.445 0.318 0.432 0.329 0.441 0.325 0.456 0.314 0.452
```

### Class mapping
```python
YOLO_CLASSES = {
    0: 'acne',
    1: 'hyperpigmentation',
    2: 'dryness',
    3: 'redness',
    4: 'dark_circles',
    5: 'large_pores'
}
# Note: fine_lines, oiliness, uneven_tone are diffuse concerns
# not well-suited to instance segmentation — use heatmap instead
```

### Training config
```yaml
# configs/yolov8seg.yaml
model: yolov8m-seg.pt
data: configs/skin_seg_data.yaml

epochs: 100
imgsz: 640
batch: 16
lr0: 0.001
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
close_mosaic: 10      # Disable mosaic augmentation last 10 epochs
overlap_mask: true
mask_ratio: 4
degrees: 10.0
flipud: 0.0
fliplr: 0.5
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
```

```yaml
# configs/skin_seg_data.yaml
path: data/yolo_seg
train: images/train
val: images/val
nc: 6
names: [acne, hyperpigmentation, dryness, redness, dark_circles, large_pores]
```

### Training command
```bash
yolo segment train \
  model=yolov8m-seg.pt \
  data=configs/skin_seg_data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  project=runs/skin_seg \
  name=v1 \
  device=0
```

### SAM 2 fallback (use at launch before masks annotated)
```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_model = build_sam2("sam2_hiera_large.yaml", "checkpoints/sam2_hiera_large.pt")
predictor = SAM2ImagePredictor(sam2_model)

def localise_concern_sam2(image, concern_bbox):
    """
    Use Stage 03 bounding box estimate as SAM2 prompt.
    concern_bbox: [x1, y1, x2, y2] normalised 0-1
    """
    predictor.set_image(image)
    h, w = image.shape[:2]
    box = np.array([
        concern_bbox[0]*w, concern_bbox[1]*h,
        concern_bbox[2]*w, concern_bbox[3]*h
    ])
    masks, scores, _ = predictor.predict(box=box, multimask_output=False)
    return masks[0], scores[0]
```

---

## Stage 05 — Severity Regression Head

Shares the EfficientNetV2-S backbone with Stage 03. No second forward pass required.

```python
class SeverityHead(nn.Module):
    """Regression head per concern. Predicts 0-100 severity score."""
    def __init__(self, feature_dim=1280, n_concerns=9):
        super().__init__()
        self.heads = nn.ModuleDict({
            concern: nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()  # Output 0-1, scaled to 0-100
            )
            for concern in CONCERN_LABELS
        })

    def forward(self, features, concern_flags):
        scores = {}
        for concern in concern_flags:
            scores[concern] = self.heads[concern](features).squeeze(-1) * 100
        return scores

class CombinedModel(nn.Module):
    """Unified model for Stages 03 + 05."""
    def __init__(self):
        super().__init__()
        backbone = timm.create_model('tf_efficientnetv2_s', pretrained=True,
                                      num_classes=0, global_pool='avg')
        self.concern_model = SkinConcernModel(backbone)
        self.severity_head = SeverityHead()

    def forward(self, x):
        logits, features = self.concern_model(x)
        concern_probs = torch.sigmoid(logits)
        concern_flags = [CONCERN_LABELS[i] for i in
                         (concern_probs > 0.5).nonzero(as_tuple=True)[1]]
        severity = self.severity_head(features, concern_flags)
        return concern_probs, severity
```

---

## Stage 07 — Progress Comparison (Longitudinal)

### Score delta (primary method)
```python
def compute_progress(
    baseline: dict,   # {concern: severity_score} from scan N-1
    current: dict     # {concern: severity_score} from current scan
) -> dict:
    progress = {}
    for concern in baseline:
        if concern not in current:
            continue
        delta = baseline[concern] - current[concern]
        pct_change = (delta / baseline[concern]) * 100 if baseline[concern] > 0 else 0
        progress[concern] = {
            'baseline': baseline[concern],
            'current': current[concern],
            'delta': round(delta, 1),
            'pct_improvement': round(pct_change, 1),
            'trend': 'improving' if delta > 5 else 'worsening' if delta < -5 else 'stable'
        }
    return progress
```

### Example progress output
```json
{
  "acne": {
    "baseline": 57,
    "current": 34,
    "delta": 23.0,
    "pct_improvement": 40.4,
    "trend": "improving"
  },
  "dryness": {
    "baseline": 44,
    "current": 51,
    "delta": -7.0,
    "pct_improvement": -15.9,
    "trend": "worsening"
  }
}
```

---

## Model registry

Use MLflow for all experiment tracking. Every model version that goes to Triton must be registered.

```python
import mlflow
import mlflow.pytorch

mlflow.set_experiment("skincare_cv")

with mlflow.start_run(run_name="efficientnetv2s_v1.2"):
    mlflow.log_params({
        "backbone": "tf_efficientnetv2_s",
        "n_concerns": 9,
        "batch_size": 32,
        "lr": 0.0001,
        "datasets": "acne04,scin,sd198"
    })
    mlflow.log_metrics({
        "val_mean_auc": 0.812,
        "val_acne_auc": 0.871,
        "fitzpatrick_v_vi_auc": 0.743
    })
    mlflow.pytorch.log_model(model, "model",
                              registered_model_name="skin_concern_detector")
```

---

## ONNX export (for Triton)

```python
def export_to_onnx(model, output_path, input_size=(1, 3, 384, 384)):
    model.eval()
    dummy_input = torch.randn(*input_size)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['concern_logits', 'severity_scores'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'concern_logits': {0: 'batch_size'},
        }
    )
    print(f"Exported to {output_path}")

export_to_onnx(combined_model, "models/skin_concern_v1.onnx")
```
