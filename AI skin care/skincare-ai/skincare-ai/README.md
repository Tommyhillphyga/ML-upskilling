# Skincare AI Platform — Technical Specification

> **Scope:** AI/CV subsystem only. This covers everything from face scan ingestion to routine generation output.  
> **Owner:** ML Lead (you) + ML Engineer  
> **Status:** Pre-build · v0.1

---

## What this system does

A user opens the app, scans their face, and within seconds receives:
- A visual skin map showing where concerns are detected and how severe they are
- A ranked list of active ingredients their skin needs
- A dermatologist-validated morning and night routine
- At re-scan (3–6 months later): a progress report comparing current vs baseline

This spec covers every component required to make that happen — models, data, schemas, infrastructure, and deployment.

---

## Spec index

| File | What it covers |
|---|---|
| [01_pipeline.md](./01_pipeline.md) | End-to-end system flow, latency budget, stage I/O |
| [02_datasets.md](./02_datasets.md) | Every dataset used, download instructions, structure, examples |
| [03_models.md](./03_models.md) | Architecture per stage, pretrained weights, training config, expected metrics |
| [04_annotation.md](./04_annotation.md) | Labelling schema, tooling, QA process, what to annotate and how |
| [05_llm_rag.md](./05_llm_rag.md) | LLM/RAG layer: knowledge base, prompt templates, example I/O |
| [06_schemas.md](./06_schemas.md) | All data contracts: API request/response, model I/O, DB tables |
| [07_infrastructure.md](./07_infrastructure.md) | Triton config, GPU requirements, ONNX export, mobile inference, storage |
| [08_evaluation.md](./08_evaluation.md) | Metrics per model, bias testing across skin tones, LLM evaluation rubric |
| [09_deployment.md](./09_deployment.md) | Phase plan, CI/CD, staging, monitoring, rollout strategy |

---

## Team and responsibilities

| Role | Responsibility |
|---|---|
| ML Lead (supervisor) | Architecture decisions, code review, Triton serving config, evaluation sign-off, client communication |
| ML Engineer | Day-to-day model training, dataset prep, pipeline implementation, experiment tracking |
| Dermatologist partner | Knowledge base authoring, annotation QA, routine template validation |
| Client team | Data annotation execution, compute provisioning, app backend integration |

---

## System at a glance

```
Mobile App
    │
    │  JPEG (face photo, ≥1080p)
    ▼
[Stage 01] Face detection & alignment      ← on-device (MediaPipe)
    │
    │  512×512 normalised face crop
    ▼
[Stage 02] Face region parsing             ← server (BiSeNetV2)
    │
    │  Zone mask JSON
    ▼
[Stage 03] Multi-label concern detection   ← server (EfficientNetV2-S)
    │
    │  Concern probability scores
    ▼
[Stage 04] Spatial localisation            ← server (YOLOv8-seg)
    │
    │  Per-concern pixel masks
    ▼
[Stage 05] Severity scoring                ← server (regression head)
    │
    │  SeverityReport JSON
    ▼
[Stage 06] Concern → ingredient mapping    ← rule engine (deterministic)
    │
    │  IngredientList JSON
    ▼
[Stage 07] LLM routine generation          ← API (Claude / GPT-4o + RAG)
    │
    │  RoutineResponse JSON
    ▼
Mobile App
```

---

## Phases

### Phase 0 — Setup (Weeks 1–2)
- Repository structure, experiment tracking (MLflow), cloud storage
- Dataset download and preprocessing scripts
- Annotation tool setup (Label Studio)
- Triton server provisioned

### Phase 1 — CV pipeline v1 (Months 1–3)
- Stages 01–02: pretrained, integrate and test
- Stage 03: fine-tune on public datasets
- Stage 04: first localisation model (SAM 2 zero-shot → YOLOv8-seg once masks annotated)
- Stage 05: severity regression head
- Internal evaluation report

### Phase 2 — LLM layer + free trial (Months 3–5)
- Stage 06: rule engine built with dermatologist
- Stage 07: LLM/RAG knowledge base + prompt templates
- Free trial: 200–300 users, collect and annotate scans
- Model v2: fine-tune on trial data

### Phase 3 — Progress tracking + paid launch prep (Months 5–7)
- Stage 07 progress comparison
- Fitzpatrick bias evaluation and model correction
- Triton production config, load testing
- Integration handoff to app backend

### Phase 4 — Post-launch iteration (Month 7+)
- User feedback loop → model retraining schedule
- A/B testing framework for routine recommendations
- Routine personalisation improvements

---

## Non-negotiables before any model goes to users

- [ ] Fitzpatrick I–VI evaluation completed (see [08_evaluation.md](./08_evaluation.md))
- [ ] Dermatologist has reviewed and signed off on routine templates
- [ ] Ingredient conflict checking validated against known interactions
- [ ] Adverse reaction escalation path defined and tested
- [ ] Facial data encryption in transit confirmed
- [ ] Model versioning and rollback tested
