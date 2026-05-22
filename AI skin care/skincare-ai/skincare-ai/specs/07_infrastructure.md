# 07 — Infrastructure

Everything required to run this system in production. Covers Triton serving, GPU requirements, model repository structure, mobile export, storage, and networking.

---

## Architecture overview

```
Mobile App
    │
    │ HTTPS (443)
    ▼
API Gateway (nginx)
    │
    ├──→ App Server (FastAPI)          — Python, handles auth, orchestration, DB
    │         │
    │         ├──→ Triton Server       — GPU inference (Stages 02–05)
    │         ├──→ LLM API             — Claude / OpenAI (Stage 07)
    │         ├──→ Pinecone            — RAG vector store
    │         └──→ PostgreSQL          — User data, scan history, routines
    │
    └──→ S3 / Object Store             — Face crops, masks, model artifacts
```

---

## GPU requirements

### Development / training
```
Recommended: NVIDIA A100 40GB (or 2× A10G 24GB)
Minimum: NVIDIA T4 16GB

For training EfficientNetV2-S (batch=32, 384px):
  - A100: ~2 hours per 50-epoch run
  - T4:   ~8 hours per 50-epoch run

For training YOLOv8-seg (batch=16, 640px):
  - A100: ~3 hours per 100-epoch run
  - T4:   ~10 hours per 100-epoch run

Cloud recommendations:
  AWS: p3.2xlarge (V100) for training, g4dn.xlarge (T4) for dev
  GCP: a2-highgpu-1g (A100) for training, n1-standard-4 + T4 for dev
```

### Production inference (Triton)
```
Recommended: NVIDIA A10G 24GB (1 per Triton instance)
  - Handles Stages 02–05 ensemble
  - Throughput: ~15 concurrent scans/second
  - Cost: ~$0.75/hour (AWS g5.xlarge)

Minimum: NVIDIA T4 16GB
  - Throughput: ~5 concurrent scans/second
  - Cost: ~$0.53/hour (AWS g4dn.xlarge)

Scale horizontally (add Triton instances) as user count grows.
At launch: 1 instance is sufficient for free trial + initial paid users.
```

---

## Triton Inference Server setup

### Model repository structure
```
triton_model_repo/
├── bisenetv2/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
├── skin_concern_detector/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
├── yolov8_seg/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
└── skin_pipeline/             ← Ensemble: runs 02→03→04→05 in sequence
    ├── config.pbtxt
    └── 1/
        └── (no model file — ensemble only)
```

### BiSeNetV2 config.pbtxt
```protobuf
name: "bisenetv2"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [3, 512, 512]
  }
]
output [
  {
    name: "zone_mask"
    data_type: TYPE_INT64
    dims: [512, 512]
  }
]
dynamic_batching {
  preferred_batch_size: [1, 4, 8]
  max_queue_delay_microseconds: 100
}
instance_group [
  { count: 1 kind: KIND_GPU }
]
optimization {
  execution_accelerators {
    gpu_execution_accelerator [
      { name: "tensorrt"
        parameters { key: "precision_mode" value: "FP16" } }
    ]
  }
}
```

### Skin concern detector config.pbtxt
```protobuf
name: "skin_concern_detector"
platform: "onnxruntime_onnx"
max_batch_size: 16
input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [3, 384, 384]
  }
]
output [
  {
    name: "concern_logits"
    data_type: TYPE_FP32
    dims: [9]
  },
  {
    name: "feature_vector"
    data_type: TYPE_FP32
    dims: [1280]
  }
]
dynamic_batching {
  preferred_batch_size: [1, 8, 16]
  max_queue_delay_microseconds: 50
}
instance_group [
  { count: 1 kind: KIND_GPU }
]
optimization {
  execution_accelerators {
    gpu_execution_accelerator [
      { name: "tensorrt"
        parameters { key: "precision_mode" value: "FP16" } }
    ]
  }
}
```

### Ensemble pipeline config.pbtxt
```protobuf
name: "skin_pipeline"
platform: "ensemble"
max_batch_size: 8

input [
  { name: "face_crop_512" data_type: TYPE_FP32 dims: [3, 512, 512] },
  { name: "face_crop_384" data_type: TYPE_FP32 dims: [3, 384, 384] }
]
output [
  { name: "zone_mask"       data_type: TYPE_INT64 dims: [512, 512] },
  { name: "concern_logits"  data_type: TYPE_FP32  dims: [9] },
  { name: "feature_vector"  data_type: TYPE_FP32  dims: [1280] }
]

ensemble_scheduling {
  step [
    {
      model_name: "bisenetv2"
      model_version: -1
      input_map  { key: "image"     value: "face_crop_512" }
      output_map { key: "zone_mask" value: "zone_mask" }
    },
    {
      model_name: "skin_concern_detector"
      model_version: -1
      input_map  { key: "image"          value: "face_crop_384" }
      output_map { key: "concern_logits" value: "concern_logits"
                   key: "feature_vector" value: "feature_vector" }
    }
  ]
}
```

### Start Triton
```bash
docker run --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /path/to/triton_model_repo:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver \
    --model-repository=/models \
    --log-verbose=1 \
    --metrics-port=8002
```

---

## App server (FastAPI)

### Key endpoints implementation pattern
```python
from fastapi import FastAPI, BackgroundTasks
import tritonclient.grpc as grpcclient
import numpy as np
import uuid

app = FastAPI()
triton = grpcclient.InferenceServerClient("triton:8001")

@app.post("/api/v1/scan")
async def submit_scan(request: ScanRequest, background_tasks: BackgroundTasks):
    scan_id = str(uuid.uuid4())

    # Save raw scan to DB (status=pending)
    await db.save_scan(scan_id, request.user_id, status="pending")

    # Process asynchronously — return scan_id immediately
    background_tasks.add_task(process_scan, scan_id, request)

    return {"scan_id": scan_id, "status": "processing",
            "poll_url": f"/api/v1/scan/{scan_id}"}

async def process_scan(scan_id: str, request: ScanRequest):
    try:
        await db.update_scan_status(scan_id, "processing")

        # Decode and preprocess face crop
        image = decode_base64_image(request.face_crop)
        image_512 = preprocess(image, size=512)
        image_384 = preprocess(image, size=384)

        # Call Triton ensemble
        inputs = [
            grpcclient.InferInput("face_crop_512", image_512.shape, "FP32"),
            grpcclient.InferInput("face_crop_384", image_384.shape, "FP32"),
        ]
        inputs[0].set_data_from_numpy(image_512)
        inputs[1].set_data_from_numpy(image_384)

        outputs = [
            grpcclient.InferRequestedOutput("zone_mask"),
            grpcclient.InferRequestedOutput("concern_logits"),
            grpcclient.InferRequestedOutput("feature_vector"),
        ]

        result = triton.infer("skin_pipeline", inputs, outputs=outputs)

        zone_mask = result.as_numpy("zone_mask")
        concern_logits = result.as_numpy("concern_logits")
        feature_vector = result.as_numpy("feature_vector")

        # Post-process
        concern_probs = sigmoid(concern_logits)
        concern_flags = get_concern_flags(concern_probs)
        severity_scores = compute_severity(feature_vector, concern_flags)
        zone_concern_map = compute_zone_concern_map(zone_mask, concern_flags)

        # Stage 06: ingredient mapping (no GPU)
        ingredient_list = map_concerns_to_ingredients(
            concern_flags, severity_scores,
            questionnaire=request.questionnaire
        )

        # Stage 07: LLM
        skin_report = await generate_skin_report(severity_scores, zone_concern_map)
        routine = await generate_routine(ingredient_list, request.questionnaire)

        # Save results
        await db.update_scan(scan_id, {
            "status": "complete",
            "severity_report": severity_scores,
            "zone_map": zone_concern_map,
            "ingredient_recommendations": ingredient_list,
            "skin_report": skin_report,
            "routine": routine
        })

    except Exception as e:
        await db.update_scan_status(scan_id, "failed")
        logger.error(f"Scan {scan_id} failed: {e}")
```

---

## Mobile inference (Stage 01 — on-device)

### iOS (CoreML via MediaPipe)
```bash
# MediaPipe is distributed with CoreML support on iOS
# Install via CocoaPods:
pod 'MediaPipeTasksVision', '~> 0.10'
```

```swift
import MediaPipeTasksVision

let faceLandmarker = try FaceLandmarker(options: FaceLandmarkerOptions(
    baseOptions: BaseOptions(modelAssetPath: "face_landmarker.task"),
    runningMode: .image,
    numFaces: 1
))

func processFrame(_ image: UIImage) -> FaceCrop? {
    let mpImage = try MPImage(uiImage: image)
    let result = try faceLandmarker.detect(image: mpImage)
    guard let landmarks = result.faceLandmarks.first else { return nil }
    return alignAndCrop(image, landmarks: landmarks)
}
```

### Android (MediaPipe Task API)
```kotlin
val options = FaceLandmarkerOptions.builder()
    .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_landmarker.task").build())
    .setNumFaces(1)
    .setRunningMode(RunningMode.IMAGE)
    .build()

val faceLandmarker = FaceLandmarker.createFromOptions(context, options)
```

---

## Storage

```
S3 Bucket structure:
skincare-prod/
├── scans/
│   └── {user_id}/{scan_id}/
│       ├── face_crop.jpg          # Original normalised crop (encrypted)
│       ├── zone_mask.png          # Stage 02 output
│       └── concern_overlay.png    # Stage 04 output (user-facing visual)
├── models/
│   └── {model_name}/{version}/
│       ├── model.onnx
│       └── metadata.json
└── knowledge_base/
    └── (RAG documents — not user data)
```

**Encryption:** All face crops encrypted at rest (AES-256, AWS SSE-S3). Access via pre-signed URLs with 1-hour expiry — never expose direct S3 URLs to clients.

**Retention:** Face crops deleted after 12 months of account inactivity, or immediately on user deletion request (GDPR/NDPR compliance).

---

## Monitoring

```python
# Key metrics to track (Prometheus + Grafana)
METRICS = {
    # Latency
    "scan_processing_time_seconds": Histogram,
    "triton_inference_latency_ms": Histogram,
    "llm_generation_time_seconds": Histogram,

    # Quality
    "scan_quality_score": Histogram,          # Distribution of quality scores
    "scan_rejection_rate": Counter,           # Quality gating rejections
    "model_confidence_distribution": Histogram,

    # Business
    "scans_per_hour": Counter,
    "routine_completion_rate": Gauge,         # % users who complete Day 1
    "progress_scan_rate": Gauge,              # % users who return for re-scan

    # Errors
    "scan_failure_rate": Counter,
    "llm_api_errors": Counter,
    "triton_unavailable_count": Counter,
}
```

### Alerting rules
```yaml
alerts:
  - name: ScanFailureRateHigh
    condition: scan_failure_rate > 0.05   # >5% scans failing
    severity: critical

  - name: LLMLatencyHigh
    condition: p95(llm_generation_time_seconds) > 8
    severity: warning

  - name: TritonDown
    condition: triton_unavailable_count > 0
    severity: critical
    notify: pagerduty

  - name: LowQualityScoreSpike
    condition: avg(scan_quality_score) < 0.7
    severity: warning
    # Could indicate mobile SDK issue or annotation drift
```
