# 06 — Schemas

All data contracts in the system. Every component communicates through these schemas. If it crosses a service boundary, it's defined here.

---

## API Endpoints

```
POST /api/v1/scan              → Submit a face scan
GET  /api/v1/scan/{scan_id}    → Get scan results
GET  /api/v1/user/{user_id}/history   → Get scan history
POST /api/v1/routine/select    → User confirms product selections
GET  /api/v1/routine/{routine_id}     → Get full routine
POST /api/v1/progress/{user_id}       → Trigger progress comparison
POST /api/v1/chat              → In-app Q&A (RAG-backed)
```

---

## 1. Scan Request

`POST /api/v1/scan`

```json
{
  "user_id": "usr_a1b2c3d4",
  "scan_type": "initial | followup",
  "face_crop": "<base64 JPEG string>",
  "face_crop_metadata": {
    "head_pose": { "yaw": 2.1, "pitch": -0.8, "roll": 0.3 },
    "quality_score": 0.94,
    "capture_timestamp": "2024-03-18T09:22:31Z",
    "device_model": "iPhone 15 Pro"
  },
  "questionnaire": {
    "skin_type_self_reported": "combination | dry | oily | normal | sensitive",
    "concerns_self_reported": ["acne", "dryness"],
    "allergies": ["fragrance", "parabens"],
    "pregnancy": false,
    "current_medications": ["tretinoin"],
    "existing_products": [
      {
        "product_name": "CeraVe PM Moisturiser",
        "step": "moisturiser",
        "routine": "night"
      }
    ]
  },
  "fitzpatrick_self_reported": 4
}
```

---

## 2. Scan Response

`GET /api/v1/scan/{scan_id}`

```json
{
  "scan_id": "scan_20240318_u1042",
  "user_id": "usr_a1b2c3d4",
  "status": "pending | processing | complete | failed",
  "created_at": "2024-03-18T09:22:31Z",
  "completed_at": "2024-03-18T09:22:34Z",

  "skin_report": {
    "headline": "Your skin is mostly balanced with a few areas to work on.",
    "overall_skin_score": 68,
    "overall_skin_score_label": "Good",
    "concerns": [
      {
        "concern": "acne",
        "headline": "Some breakout activity",
        "explanation": "We detected moderate breakout activity concentrated around your chin and T-zone.",
        "severity_label": "Moderate",
        "severity_score": 57,
        "zones": ["chin", "t_zone"]
      }
    ]
  },

  "severity_report": {
    "concern_scores": {
      "acne": { "raw": 2.3, "normalised": 57, "label": "Moderate" },
      "hyperpigmentation": { "raw": 0.8, "normalised": 20, "label": "Mild" },
      "dryness": { "raw": 1.8, "normalised": 44, "label": "Moderate" },
      "oiliness": { "raw": 0.4, "normalised": 10, "label": "Minimal" },
      "redness": { "raw": 0.3, "normalised": 7, "label": "Minimal" },
      "dark_circles": { "raw": 2.9, "normalised": 72, "label": "Significant" },
      "fine_lines": { "raw": 0.2, "normalised": 5, "label": "Minimal" },
      "large_pores": { "raw": 1.2, "normalised": 29, "label": "Mild" },
      "uneven_tone": { "raw": 0.9, "normalised": 22, "label": "Mild" }
    },
    "concern_flags": ["acne", "dryness", "dark_circles"],
    "model_version": "efficientnetv2s_v1.2"
  },

  "zone_map": {
    "zone_mask_url": "https://cdn.skincare.app/masks/scan_20240318_u1042_zones.png",
    "concern_overlay_url": "https://cdn.skincare.app/masks/scan_20240318_u1042_concerns.png",
    "concern_zone_map": {
      "acne": ["chin", "t_zone"],
      "dryness": ["left_cheek", "right_cheek"],
      "dark_circles": ["eye_area"]
    }
  },

  "ingredient_recommendations": {
    "prioritised": [
      {
        "rank": 1,
        "ingredient": "niacinamide",
        "targets": ["acne", "dark_circles", "large_pores"],
        "rationale": "Addresses three of your top concerns. Well-tolerated by combination skin.",
        "concentration_range": "4-10%",
        "routine_step": "moisturiser",
        "am": true,
        "pm": true
      },
      {
        "rank": 2,
        "ingredient": "salicylic_acid",
        "targets": ["acne", "large_pores"],
        "rationale": "BHA that penetrates pores to clear congestion at the source.",
        "concentration_range": "1-2%",
        "routine_step": "cleanser",
        "am": true,
        "pm": true
      },
      {
        "rank": 3,
        "ingredient": "retinol",
        "targets": ["dark_circles", "fine_lines"],
        "rationale": "Builds collagen and reduces periorbital pigmentation over time.",
        "concentration_range": "0.025-0.05% (start low)",
        "routine_step": "treatment",
        "am": false,
        "pm": true,
        "cautions": ["avoid_during_pregnancy", "introduce_slowly"]
      }
    ],
    "avoid": [
      {
        "ingredient": "fragrance",
        "reason": "User-reported allergy"
      }
    ]
  }
}
```

---

## 3. Routine Response

`GET /api/v1/routine/{routine_id}`

```json
{
  "routine_id": "routine_20240318_u1042",
  "scan_id": "scan_20240318_u1042",
  "user_id": "usr_a1b2c3d4",
  "created_at": "2024-03-18T09:23:05Z",
  "version": 1,
  "morning": [
    {
      "step": 1,
      "step_name": "Cleanse",
      "product_type": "Salicylic acid cleanser",
      "key_ingredient": "salicylic_acid",
      "instruction": "Wet your face, apply a small amount, massage for 60 seconds. Rinse with lukewarm water.",
      "why": "Salicylic acid penetrates your pores to prevent the congestion that leads to breakouts.",
      "user_product": null,
      "what_to_look_for": "1-2% salicylic acid. BHA on the label means the same thing.",
      "frequency": "daily",
      "duration_seconds": 60,
      "video_tutorial_url": "https://cdn.skincare.app/tutorials/cleanse_basics.mp4"
    },
    {
      "step": 2,
      "step_name": "Moisturise",
      "product_type": "Lightweight gel moisturiser with niacinamide",
      "key_ingredient": "niacinamide",
      "instruction": "Apply a pea-sized amount over the whole face.",
      "why": "Niacinamide regulates oil in your T-zone and chin while keeping your cheeks hydrated.",
      "user_product": {
        "product_name": "CeraVe PM Moisturising Lotion",
        "already_owned": true
      },
      "what_to_look_for": null,
      "frequency": "daily",
      "duration_seconds": 30,
      "video_tutorial_url": null
    },
    {
      "step": 3,
      "step_name": "SPF",
      "product_type": "Mineral sunscreen SPF 30+",
      "key_ingredient": "zinc_oxide",
      "instruction": "Apply generously. This is the most important step in your morning routine.",
      "why": "UV exposure worsens acne scarring and dark circles. SPF protects everything else you're doing.",
      "user_product": null,
      "what_to_look_for": "SPF 30 or higher. Zinc oxide preferred. Avoid alcohol-heavy formulas if your skin is dry.",
      "frequency": "daily",
      "duration_seconds": 30,
      "video_tutorial_url": "https://cdn.skincare.app/tutorials/spf_application.mp4"
    }
  ],
  "night": [
    {
      "step": 1,
      "step_name": "Cleanse",
      "product_type": "Gentle cleanser",
      "key_ingredient": null,
      "instruction": "Cleanse to remove SPF and product buildup.",
      "why": "Clean skin absorbs your treatment products better.",
      "user_product": null,
      "what_to_look_for": "Gentle, fragrance-free. Your morning cleanser works here too.",
      "frequency": "daily",
      "duration_seconds": 60,
      "video_tutorial_url": null
    },
    {
      "step": 2,
      "step_name": "Eye treatment",
      "product_type": "Retinol eye cream",
      "key_ingredient": "retinol",
      "instruction": "Apply a rice-grain amount under each eye only. Do not apply to eyelid.",
      "why": "Retinol builds collagen and reduces the pigmentation we detected under your eyes.",
      "user_product": null,
      "what_to_look_for": "0.025–0.05% retinol for beginners. Avoid anything stronger to start.",
      "frequency": "2x per week initially, build to nightly over 4 weeks",
      "duration_seconds": 30,
      "video_tutorial_url": "https://cdn.skincare.app/tutorials/retinol_intro.mp4",
      "caution": "Do not use if pregnant. Some redness and flaking in week 1-2 is normal."
    },
    {
      "step": 3,
      "step_name": "Moisturise",
      "product_type": "Barrier repair moisturiser",
      "key_ingredient": "ceramides",
      "instruction": "Apply generously. On retinol nights, layer this over your eye cream.",
      "why": "Ceramides restore your skin barrier, reducing sensitivity to your active ingredients.",
      "user_product": {
        "product_name": "CeraVe PM Moisturising Lotion",
        "already_owned": true
      },
      "what_to_look_for": null,
      "frequency": "daily",
      "duration_seconds": 30,
      "video_tutorial_url": null
    }
  ],
  "conflict_warnings": [],
  "introduction_schedule": "Week 1-2: cleanser + moisturiser + SPF only. Week 3: add retinol (2x/week). Week 5: increase retinol frequency based on tolerance.",
  "notes": "Introduce one new product at a time. Wait 2 weeks before adding anything new.",
  "dermatologist_validated": true,
  "template_version": "concern_routines_v2.1"
}
```

---

## 4. Progress Report

`POST /api/v1/progress/{user_id}`

```json
{
  "user_id": "usr_a1b2c3d4",
  "baseline_scan_id": "scan_20231215_u1042",
  "current_scan_id": "scan_20240318_u1042",
  "weeks_elapsed": 13,

  "progress": {
    "overall_skin_score_baseline": 54,
    "overall_skin_score_current": 68,
    "overall_improvement": 14,
    "trend": "improving",

    "concern_progress": {
      "acne": {
        "baseline": 78,
        "current": 57,
        "delta": 21,
        "pct_improvement": 26.9,
        "trend": "improving"
      },
      "dark_circles": {
        "baseline": 80,
        "current": 72,
        "delta": 8,
        "pct_improvement": 10.0,
        "trend": "improving"
      },
      "dryness": {
        "baseline": 38,
        "current": 44,
        "delta": -6,
        "pct_improvement": -15.8,
        "trend": "worsening"
      }
    },

    "narrative": {
      "headline": "Strong progress on acne — down 27% in 13 weeks.",
      "body": "Your most significant improvement is in breakout activity, particularly on your chin and T-zone. Your salicylic acid cleanser and niacinamide routine is working. Dark circles have also improved, though retinol results typically build over 6+ months, so expect continued progress. Dryness has increased slightly — this is common during retinol introduction and usually resolves as skin adapts.",
      "recommendation": "continue"
    }
  },

  "routine_update": {
    "recommended": true,
    "reason": "Dryness has increased — moisturiser may need upgrading. Acne progress is strong — consider introducing a vitamin C serum for pigmentation.",
    "changes_suggested": [
      {
        "type": "upgrade",
        "step": "moisturiser",
        "current": "lightweight gel",
        "suggested": "richer cream with ceramides + hyaluronic acid"
      }
    ]
  }
}
```

---

## 5. Database Tables

```sql
-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    fitzpatrick_self_reported SMALLINT,
    questionnaire JSONB,
    subscription_status TEXT DEFAULT 'trial',
    subscription_expires_at TIMESTAMPTZ
);

-- Scans
CREATE TABLE scans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    scan_type TEXT CHECK (scan_type IN ('initial', 'followup')),
    created_at TIMESTAMPTZ DEFAULT now(),
    status TEXT DEFAULT 'pending',
    face_crop_s3_key TEXT,
    severity_report JSONB,
    zone_map JSONB,
    ingredient_recommendations JSONB,
    skin_report JSONB,
    model_version TEXT,
    is_baseline BOOLEAN DEFAULT false
);

-- Routines
CREATE TABLE routines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id UUID REFERENCES scans(id),
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    version SMALLINT DEFAULT 1,
    morning JSONB,
    night JSONB,
    conflict_warnings JSONB,
    dermatologist_validated BOOLEAN DEFAULT false,
    template_version TEXT
);

-- Progress reports
CREATE TABLE progress_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    baseline_scan_id UUID REFERENCES scans(id),
    current_scan_id UUID REFERENCES scans(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    progress JSONB,
    routine_update JSONB
);

-- Annotation (internal use)
CREATE TABLE scan_annotations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id UUID REFERENCES scans(id),
    annotator_id TEXT,
    annotation_timestamp TIMESTAMPTZ,
    concern_labels TEXT[],
    severity_labels JSONB,
    confidence TEXT CHECK (confidence IN ('high', 'medium', 'low_refer_to_derm')),
    dermatologist_reviewed BOOLEAN DEFAULT false,
    notes TEXT
);

-- Indexes
CREATE INDEX idx_scans_user_id ON scans(user_id);
CREATE INDEX idx_scans_created_at ON scans(created_at);
CREATE INDEX idx_routines_user_id ON routines(user_id);
```
