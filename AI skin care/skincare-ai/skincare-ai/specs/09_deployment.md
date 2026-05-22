# 09 — Deployment

Phase plan, environment setup, CI/CD, model rollout, and post-launch operations.

---

## Environments

| Environment | Purpose | GPU | LLM |
|---|---|---|---|
| `dev` | Local development, unit tests | CPU only (small batch) | mocked |
| `staging` | Integration tests, QA, dermatologist review | 1× T4 (real inference) | real (limited) |
| `prod` | Live users | 1× A10G (scaling as needed) | real |

---

## Phase plan

### Phase 0 — Setup (Weeks 1–2)

```bash
# Repository structure
skincare-ai/
├── training/
│   ├── bisenetv2/
│   ├── efficientnetv2s/
│   └── yolov8_seg/
├── serving/
│   ├── triton_model_repo/
│   └── app/                  # FastAPI app server
├── data/
│   ├── raw/                  # Downloaded datasets
│   ├── processed/            # Preprocessed, split
│   └── knowledge_base/       # Dermatologist content
├── evaluation/
├── notebooks/                # Exploratory analysis
├── configs/
├── scripts/
│   ├── download_datasets.sh
│   ├── preprocess.py
│   └── export_onnx.py
└── tests/
```

**Week 1 checklist:**
- [ ] Repository created, team access granted
- [ ] MLflow tracking server provisioned (cloud or self-hosted)
- [ ] Label Studio instance running (cloud VM or Render)
- [ ] S3 bucket created with encryption enabled
- [ ] PostgreSQL provisioned (AWS RDS or Supabase)
- [ ] Development GPU instance available
- [ ] Dataset download scripts tested (CelebAMask-HQ, SCIN, ACNE04)
- [ ] Dermatologist partner briefed on annotation requirements

**Week 2 checklist:**
- [ ] BiSeNetV2 pretrained weights downloaded and inference tested
- [ ] EfficientNetV2-S pretrained from timm confirmed running
- [ ] ONNX export pipeline working for both models
- [ ] Triton docker image confirmed on GPU
- [ ] FastAPI skeleton with `/scan` endpoint returning mocked response
- [ ] Mobile MediaPipe integration confirmed (with client team)

---

### Phase 1 — CV Pipeline v1 (Months 1–3)

**Month 1:**
- Download and preprocess all public datasets
- Fine-tune BiSeNetV2 on CelebAMask-HQ → evaluate → export ONNX → test in Triton
- Fine-tune EfficientNetV2-S on ACNE04 + SD-198 (acne-focused first pass)
- Annotation: first batch of zone masks (100 images) for BiSeNetV2 custom zones

**Month 2:**
- Add SCIN to EfficientNetV2-S training → full 9-concern model
- First Fitzpatrick evaluation report → document gaps
- YOLOv8-seg: SAM2 zero-shot baseline running in pipeline
- FastAPI `/scan` endpoint returning real CV results (skin report mocked)
- Internal milestone: full CV pipeline running end-to-end on test images

**Month 3:**
- Annotation: 300 classification labels from internal sources
- Retrain EfficientNetV2-S with internal data → v1.1
- Severity scoring head added and evaluated
- Triton ensemble pipeline (`skin_pipeline`) running stably
- Evaluation report v1 completed and reviewed

**Milestone deliverable (end of Month 3):**
```
POST /api/v1/scan → returns ScanResponse with:
  ✅ severity_report (real model output)
  ✅ zone_map (real model output)
  ✅ skin_report (LLM-generated — Phase 2 will be more tested)
  ⚠️ routine (template-based placeholder — full LLM in Phase 2)
```

---

### Phase 2 — LLM Layer + Free Trial (Months 3–5)

**Month 3–4:**
- Dermatologist authors `concern_routines.yaml` and `ingredient_conflicts.yaml`
- Knowledge base indexed in Pinecone
- Skin report prompt template developed and evaluated (50-case rubric)
- Routine generation prompt template developed and evaluated
- RAG retrieval tested against real concern profiles

**Month 4–5 — Free Trial:**
- 200–300 users invited (limited access)
- Every scan annotated (classification labels + dermatologist review on 50%)
- Prompt templates iterated based on dermatologist feedback
- Model v1.2 trained on trial scan data

**Free trial success criteria:**
```
Technical:
  - <5% scan failure rate
  - <3.5s end-to-end latency (p95)
  - >90% LLM output passes schema validation

Clinical:
  - Dermatologist approves >85% of generated routines (sample review)
  - Zero adverse reaction reports

User:
  - >70% of users complete their first routine review
  - >60% report the skin report felt accurate
```

---

### Phase 3 — Progress Tracking + Launch Prep (Months 5–7)

**Month 5–6:**
- Progress comparison pipeline (Stage 07 longitudinal)
- Progress narrative prompt template evaluated
- YOLOv8-seg training: first 500 annotated masks available
- Fitzpatrick re-evaluation with v1.2 model — must pass hard thresholds
- Load testing: Triton + FastAPI under simulated concurrent load

**Month 6–7:**
- Integration handoff to app backend team (API contracts locked from [06_schemas.md](./06_schemas.md))
- Monitoring dashboards live (Grafana + Prometheus)
- Rollback procedure tested
- Security review: face data encryption, access control, data retention
- Soft launch: internal team + waitlist users (~50 paid)

**Month 7 — Paid Launch:**
- Full paid subscription available
- Monitoring on 24h rotation for first 2 weeks
- Model retraining schedule confirmed (monthly, triggered by data accumulation)

---

## CI/CD pipeline

```yaml
# .github/workflows/ml_pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main, staging]
  pull_request:
    branches: [main]

jobs:
  unit_tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest tests/unit/ -v

  schema_validation:
    runs-on: ubuntu-latest
    steps:
      - name: Validate all JSON schemas
        run: python scripts/validate_schemas.py

  model_evaluation:
    runs-on: [self-hosted, gpu]   # Runs on GPU runner
    if: github.ref == 'refs/heads/staging'
    steps:
      - name: Run evaluation suite
        run: python evaluation/run_all_evals.py --env staging
      - name: Check pass thresholds
        run: python evaluation/check_thresholds.py
      - name: Publish evaluation report
        run: python evaluation/publish_report.py --to mlflow

  triton_smoke_test:
    runs-on: [self-hosted, gpu]
    needs: model_evaluation
    steps:
      - name: Start Triton (test mode)
        run: docker-compose -f docker-compose.test.yml up -d triton
      - name: Run inference smoke tests
        run: pytest tests/integration/test_triton.py -v
      - name: Teardown
        run: docker-compose -f docker-compose.test.yml down

  deploy_staging:
    needs: [triton_smoke_test]
    if: github.ref == 'refs/heads/staging'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: ./scripts/deploy.sh staging

  deploy_prod:
    needs: [triton_smoke_test]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production    # Requires manual approval in GitHub
    steps:
      - name: Deploy to production
        run: ./scripts/deploy.sh prod
```

---

## Model versioning and rollback

```python
# Every production model is tagged in MLflow and S3
# Triton loads models by version — rolling back = pointing to previous version

def deploy_model(model_name: str, version: str, env: str = "prod"):
    """Swap a model version in Triton without downtime."""
    # Copy ONNX to model repo
    s3.copy(
        f"models/{model_name}/{version}/model.onnx",
        f"triton_model_repo/{model_name}/{version}/model.onnx"
    )
    # Update Triton config to use new version
    update_triton_config(model_name, version=int(version))
    # Triton hot-reloads on config change — no restart needed
    triton_client.load_model(model_name)
    print(f"Deployed {model_name} v{version} to {env}")

def rollback_model(model_name: str):
    """Roll back to the previous stable version."""
    prev_version = get_previous_stable_version(model_name)
    deploy_model(model_name, prev_version)
    print(f"Rolled back {model_name} to v{prev_version}")
```

---

## Post-launch retraining schedule

| Trigger | Action |
|---|---|
| 500 new annotated scans accumulated | Retrain EfficientNetV2-S, evaluate, stage for deployment |
| New concern AUC drops >3% vs baseline (monitoring alert) | Investigate + emergency retrain |
| 3-month mark (subscription renewal cycle) | Full evaluation report, model update if warranted |
| Dermatologist flags systematic error in routines | Prompt template update + re-evaluation |
| New adverse reaction reported | Halt affected routine template, investigate, dermatologist review |

---

## Handoff checklist (to client team)

Before the AI system is integrated with the app:

- [ ] API spec locked ([06_schemas.md](./06_schemas.md)) — no schema changes without versioning
- [ ] Staging environment access granted to client backend team
- [ ] Error response format agreed (`{"error": "...", "code": "..."}`)
- [ ] Rate limits agreed (requests per user per day)
- [ ] SLA agreed (99.5% uptime, <3.5s p95 latency)
- [ ] Data deletion endpoint implemented and tested (`DELETE /api/v1/user/{user_id}`)
- [ ] Monitoring dashboard access granted to client
- [ ] Escalation path for adverse reactions defined in writing
