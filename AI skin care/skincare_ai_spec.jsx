import { useState } from "react";

const TABS = ["Pipeline", "Models", "Datasets", "Open Questions"];

const pipeline = [
  {
    stage: "01",
    title: "Face detection & alignment",
    color: "c-teal",
    inputs: "Raw camera frame",
    outputs: "Cropped, normalised face (224×224 or 384×384)",
    notes: "On-device. Runs before any data leaves the phone.",
  },
  {
    stage: "02",
    title: "Face region parsing",
    color: "c-teal",
    inputs: "Normalised face image",
    outputs: "Semantic zone mask (forehead, cheeks, nose, chin, T-zone, eye area)",
    notes: "Maps concern locations to named facial regions for the visual skin map.",
  },
  {
    stage: "03",
    title: "Multi-label concern detection",
    color: "c-purple",
    inputs: "Normalised face image",
    outputs: "Per-concern probability scores [0–1]",
    notes: "Identifies: acne, hyperpigmentation, dryness, oiliness, redness, dark circles, fine lines, large pores, uneven tone.",
  },
  {
    stage: "04",
    title: "Spatial concern localisation",
    color: "c-purple",
    inputs: "Face image + Stage 03 concern flags",
    outputs: "Pixel-level masks per concern → per-zone severity overlay",
    notes: "Produces the visual skin map the user sees.",
  },
  {
    stage: "05",
    title: "Severity scoring",
    color: "c-purple",
    inputs: "Concern masks",
    outputs: "0–100 severity score per concern per zone",
    notes: "Stored as the baseline for progress tracking. Acne uses IGA 0–4 scale internally.",
  },
  {
    stage: "06",
    title: "Concern → ingredient mapping",
    color: "c-amber",
    inputs: "Ranked concern list + questionnaire (allergies, sensitivity)",
    outputs: "Prioritised active ingredient list",
    notes: "Dermatologist-validated rule engine. Not a trained model — deterministic and auditable.",
  },
  {
    stage: "07",
    title: "Progress comparison",
    color: "c-blue",
    inputs: "Current severity scores + historical baseline",
    outputs: "ΔSeverity per concern per zone + trend direction",
    notes: "Triggers routine update recommendations at 3/6-month re-scan.",
  },
];

const models = [
  {
    stage: "Stage 01",
    name: "MediaPipe Face Mesh",
    task: "Face detection & 468-point landmark alignment",
    arch: "BlazeFace (detector) + facial landmark model",
    input: "RGB frame, any resolution",
    output: "468 3D landmarks, bounding box, head pose",
    inference: "On-device (TFLite / iOS CoreML)",
    why: "Sub-5ms on mobile, no network call, no facial data sent server-side. Handles roll/pitch/yaw up to ~30°. Google-maintained.",
    alt: "RetinaFace (server-side, higher accuracy for difficult angles); MTCNN (older, heavier)",
    color: "teal",
  },
  {
    stage: "Stage 02",
    name: "BiSeNetV2 (face-parsing.PyTorch)",
    task: "Semantic face region segmentation",
    arch: "BiSeNetV2 backbone, pretrained on CelebAMask-HQ (19 classes)",
    input: "512×512 RGB face",
    output: "Per-pixel class label: skin, hair, l/r eye, nose, lips, ears, neck, background",
    inference: "Server-side (GPU). ~15ms on A10.",
    why: "Lightweight, fast, open weights. CelebAMask-HQ covers diverse faces. Easily fine-tuned to add zone subdivisions (T-zone, cheek zones).",
    alt: "SegFormer-B2 (better accuracy, heavier); DML-CSR (SOTA face parsing)",
    color: "teal",
  },
  {
    stage: "Stage 03",
    name: "EfficientNetV2-S (fine-tuned)",
    task: "Multi-label skin concern classification",
    arch: "EfficientNetV2-S backbone + multi-label head (sigmoid per concern). ImageNet pretrained → skin domain fine-tuned.",
    input: "384×384 RGB face",
    output: "9× sigmoid scores: [acne, pigmentation, dryness, oiliness, redness, dark circles, fine lines, pores, uneven tone]",
    inference: "Server-side. ~8ms on T4.",
    why: "Strong ImageNet features transfer well to texture/tone tasks. Compact enough to iterate quickly. Multi-label BCE loss handles co-occurring concerns.",
    alt: "ViT-B/16 (better on subtle tone issues, needs more data); ResNet-50 (baseline, weaker on fine-grained texture)",
    color: "purple",
  },
  {
    stage: "Stage 04",
    name: "YOLOv8-seg (fine-tuned) or SAM 2 (prompted)",
    task: "Pixel-level spatial concern localisation",
    arch: "Option A — YOLOv8-seg: instance segmentation model fine-tuned on labelled concern masks. Option B — SAM 2 with Stage 03 concern detections as prompt boxes.",
    input: "384×384 face + Stage 03 concern flags",
    output: "Binary mask per concern + bounding polygon",
    inference: "Server-side. YOLOv8-seg ~20ms; SAM 2 ~50ms on A10.",
    why: "YOLOv8-seg: best if you can annotate training masks (higher accuracy, deterministic). SAM 2: strong zero-shot option if annotation budget is limited — Stage 03 boxes prompt it.",
    alt: "Mask R-CNN (heavier, slower); DeepLabV3+ (semantic only, not instance)",
    color: "purple",
  },
  {
    stage: "Stage 05",
    name: "Regression head (on EfficientNetV2-S features)",
    task: "Severity scoring per concern",
    arch: "Shared backbone from Stage 03 + lightweight regression MLP head. Acne: IGA 0–4 ordinal regression. Others: 0–100 continuous score.",
    input: "Stage 03 features + concern masks from Stage 04",
    output: "Scalar severity score per concern",
    inference: "Piggybacks Stage 03 forward pass — near-zero added cost.",
    why: "Reusing Stage 03 features avoids a second forward pass. IGA scale is clinically validated for acne; maps cleanly to dermatologist review.",
    alt: "Separate regression model per concern (more accurate, 9× inference cost)",
    color: "purple",
  },
  {
    stage: "Stage 07",
    name: "Score delta + optional Siamese net",
    task: "Longitudinal progress comparison",
    arch: "Primary: direct severity score comparison (ΔScore = current − baseline). Optional enhancement: Siamese EfficientNet trained on before/after pairs to learn perceptual improvement signal independent of lighting/camera variance.",
    input: "Severity score vectors across time points",
    output: "Per-concern improvement/regression delta + confidence interval",
    inference: "Lightweight; runs at re-scan time only.",
    why: "Score delta is interpretable and dermatologist-reviewable. Siamese net adds robustness to lighting/pose drift across scans taken months apart.",
    alt: "SSIM / LPIPS (perceptual similarity — no concern-specificity)",
    color: "blue",
  },
];

const datasets = [
  {
    category: "Face detection & alignment",
    color: "teal",
    items: [
      {
        name: "FFHQ",
        size: "70,000 images",
        license: "CC-BY-NC 2.0",
        link: "https://github.com/NVlabs/ffhq-dataset",
        use: "Pretraining & domain adaptation — high quality, diverse ages/ethnicities",
        note: "Not labelled for skin concerns. Use for representation learning.",
      },
      {
        name: "CelebAMask-HQ",
        size: "30,000 images",
        license: "CC BY-NC 4.0",
        link: "https://github.com/switchablenorms/CelebAMask-HQ",
        use: "Face parsing (Stage 02) — 19 semantic region masks per image",
        note: "Primary pretraining set for BiSeNetV2.",
      },
    ],
  },
  {
    category: "Skin concern detection",
    color: "purple",
    items: [
      {
        name: "ACNE04",
        size: "1,457 images",
        license: "Research use",
        link: "https://github.com/xpwu95/LDL",
        use: "Acne detection + IGA severity grading (Grade I–IV)",
        note: "Facial photos, not dermoscopy. Most relevant to your use case.",
      },
      {
        name: "Fitzpatrick17k",
        size: "16,577 images",
        license: "CC BY 4.0",
        link: "https://github.com/mattgroh/fitzpatrick17k",
        use: "Fairness validation — covers all 6 Fitzpatrick skin tones with condition labels",
        note: "Critical. Must test your models against this. Skipping it = biased outputs.",
      },
      {
        name: "SCIN (Google)",
        size: "5,000+ images",
        license: "CC BY 4.0",
        link: "https://github.com/google-research-datasets/scin",
        use: "Consumer-style skin condition photos — closest to what your users will submit",
        note: "Diverse skin tones, shot on phones not clinical cameras. High priority.",
      },
      {
        name: "SkinCon",
        size: "3,230 images",
        license: "Research use",
        link: "https://skincon-dataset.github.io",
        use: "Fine-grained dermatology concept annotations — useful for explainability",
        note: "Annotated with clinical concepts (e.g. scaling, erythema). Aids concern localisation.",
      },
      {
        name: "SD-198",
        size: "6,584 images",
        license: "Research use",
        link: "http://xiaoxiaosun.com/docs/2016-eccv-skin.pdf",
        use: "198 skin condition categories — broad coverage for pretraining",
        note: "More clinical than cosmetic; use as auxiliary pretraining data.",
      },
      {
        name: "ISIC Archive",
        size: "50,000+ images",
        license: "CC0 / CC BY",
        link: "https://www.isic-archive.com",
        use: "Skin lesion segmentation masks — useful for localisation model pretraining",
        note: "Dermoscopy (not face photos). Transfer learning for texture segmentation only.",
      },
    ],
  },
  {
    category: "Ingredients & products",
    color: "amber",
    items: [
      {
        name: "Open Beauty Facts",
        size: "~200,000 products",
        license: "ODbL (open)",
        link: "https://world.openbeautyfacts.org",
        use: "INCI ingredient lists per product — maps products to active ingredients",
        note: "Community-maintained. Quality varies; needs cleaning. Best open source available.",
      },
      {
        name: "INCIDecoder database",
        size: "~20,000 ingredients",
        license: "Scraping terms apply",
        link: "https://incidecoder.com",
        use: "Ingredient function labels and safety ratings",
        note: "Not freely downloadable. Consider partnering or building your own via dermatologist input.",
      },
    ],
  },
  {
    category: "Synthetic & augmentation",
    color: "gray",
    items: [
      {
        name: "StyleGAN3 / Stable Diffusion (fine-tuned)",
        size: "Generatable",
        license: "Depends on base model",
        link: "",
        use: "Generate synthetic labelled training pairs — especially for rare concern combinations and underrepresented skin tones",
        note: "Use with care: synthetic acne/pigmentation images must be validated by a dermatologist before entering the training set.",
      },
      {
        name: "Lighting/pose augmentation",
        size: "Applied to all datasets",
        license: "N/A",
        link: "",
        use: "Simulate phone-camera variance: random colour jitter, white balance shifts, blur, low-light, shadows",
        note: "Critical. Your users will take photos in bathrooms under fluorescent or warm lighting — models trained only on clinical photos will fail.",
      },
    ],
  },
];

const questions = [
  {
    q: "On-device vs server-side inference?",
    detail: "Stage 01 (face detection) must be on-device — no raw biometric data should leave the phone unprocessed. Stages 02–05 can run server-side with GPU acceleration. Consider encrypting the normalised face crop in transit. Decision affects latency, cost, and your privacy policy.",
    priority: "Urgent",
  },
  {
    q: "Who annotates the training masks?",
    detail: "ACNE04 and SCIN give you concern labels, not always pixel masks. You'll need dermatologist annotators (or at minimum dermatologist review) to label spatial extents per concern, especially for hyperpigmentation vs redness vs uneven tone — they overlap significantly.",
    priority: "Urgent",
  },
  {
    q: "Fitzpatrick scale coverage in your test set?",
    detail: "Fitzpatrick17k is a validation resource, not a training set. You need to ensure your annotation pipeline deliberately recruits images across FST I–VI. Models trained predominantly on FST I–III systematically underperform on FST IV–VI for hyperpigmentation and redness detection specifically.",
    priority: "Urgent",
  },
  {
    q: "How is 'progress' quantified for the user?",
    detail: "Severity score delta is internally useful but hard to communicate. Define a user-facing progress metric early (e.g. 'Acne coverage reduced by 40%' or a 5-star improvement scale). This affects how Stage 05 and 07 outputs are normalised and displayed.",
    priority: "High",
  },
  {
    q: "Regulatory classification?",
    detail: "If the platform makes clinical-sounding claims ('treats acne', 'reduces hyperpigmentation') rather than cosmetic ones ('helps reduce the appearance of'), it may cross into medical device territory in some markets (EU MDR, UK MHRA, US FDA SaMD). Worth legal review early.",
    priority: "High",
  },
  {
    q: "How do you handle adverse reactions?",
    detail: "A user follows the recommended routine and experiences a reaction. Your AI flagged the wrong concern or missed an allergy. Define the escalation path: dermatologist review trigger, feedback loop back into the model, and liability scope.",
    priority: "High",
  },
  {
    q: "Free trial data strategy?",
    detail: "The doc mentions a limited free trial to test accuracy. This is your primary data collection opportunity. Define opt-in consent, annotation pipeline, and ground truth collection (self-report vs dermatologist review) before launch — don't waste the trial cohort.",
    priority: "Medium",
  },
];

const colorMap = {
  teal: { bg: "#E1F5EE", border: "#0F6E56", text: "#085041", tag: "#9FE1CB", tagText: "#04342C" },
  purple: { bg: "#EEEDFE", border: "#534AB7", text: "#26215C", tag: "#CECBF6", tagText: "#26215C" },
  amber: { bg: "#FAEEDA", border: "#854F0B", text: "#412402", tag: "#FAC775", tagText: "#412402" },
  blue: { bg: "#E6F1FB", border: "#185FA5", text: "#042C53", tag: "#B5D4F4", tagText: "#042C53" },
  gray: { bg: "#F1EFE8", border: "#5F5E5A", text: "#2C2C2A", tag: "#D3D1C7", tagText: "#2C2C2A" },
};

const priorityColor = { Urgent: "#A32D2D", High: "#854F0B", Medium: "#3B6D11" };
const priorityBg = { Urgent: "#FCEBEB", High: "#FAEEDA", Medium: "#EAF3DE" };

export default function Spec() {
  const [tab, setTab] = useState("Pipeline");
  const [expanded, setExpanded] = useState({});
  const toggle = (i) => setExpanded((e) => ({ ...e, [i]: !e[i] }));

  return (
    <div style={{ fontFamily: "var(--font-sans)", padding: "1.5rem 0", maxWidth: 680 }}>
      <div style={{ marginBottom: "0.25rem" }}>
        <span style={{ fontSize: 11, letterSpacing: "0.12em", color: "var(--color-text-tertiary)", textTransform: "uppercase" }}>
          Technical spec
        </span>
      </div>
      <h1 style={{ fontSize: 22, fontWeight: 500, margin: "0 0 0.25rem", color: "var(--color-text-primary)" }}>
        AI / CV architecture
      </h1>
      <p style={{ fontSize: 14, color: "var(--color-text-secondary)", margin: "0 0 1.5rem", lineHeight: 1.6 }}>
        Skincare platform · your scope
      </p>

      {/* Tab bar */}
      <div style={{ display: "flex", gap: 4, marginBottom: "1.5rem", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              background: "none",
              border: "none",
              borderBottom: tab === t ? "2px solid var(--color-text-primary)" : "2px solid transparent",
              padding: "8px 14px",
              fontSize: 14,
              fontWeight: tab === t ? 500 : 400,
              color: tab === t ? "var(--color-text-primary)" : "var(--color-text-secondary)",
              cursor: "pointer",
              marginBottom: -1,
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {/* PIPELINE TAB */}
      {tab === "Pipeline" && (
        <div>
          {pipeline.map((s, i) => {
            const c = colorMap[s.color.replace("c-", "")] || colorMap.gray;
            return (
              <div key={i} style={{ display: "flex", gap: 16, marginBottom: 12 }}>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", flexShrink: 0 }}>
                  <div style={{
                    width: 36, height: 36, borderRadius: "50%",
                    background: c.bg, border: `0.5px solid ${c.border}`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 11, fontWeight: 500, color: c.text, letterSpacing: "0.05em"
                  }}>{s.stage}</div>
                  {i < pipeline.length - 1 && (
                    <div style={{ width: 1, flex: 1, minHeight: 16, background: "var(--color-border-tertiary)", margin: "4px 0" }} />
                  )}
                </div>
                <div style={{
                  flex: 1, background: "var(--color-background-secondary)",
                  border: "0.5px solid var(--color-border-tertiary)",
                  borderRadius: "var(--border-radius-lg)",
                  padding: "14px 16px",
                  marginBottom: i < pipeline.length - 1 ? 4 : 0,
                }}>
                  <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 8, color: "var(--color-text-primary)" }}>
                    {s.title}
                  </div>
                  <div style={{ fontSize: 12, color: "var(--color-text-secondary)", lineHeight: 1.6 }}>
                    <span style={{ color: "var(--color-text-tertiary)" }}>In → </span>{s.inputs}<br />
                    <span style={{ color: "var(--color-text-tertiary)" }}>Out → </span>{s.outputs}
                  </div>
                  <div style={{
                    marginTop: 8, padding: "6px 10px",
                    background: c.bg, borderRadius: "var(--border-radius-md)",
                    fontSize: 12, color: c.text, lineHeight: 1.5
                  }}>
                    {s.notes}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* MODELS TAB */}
      {tab === "Models" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {models.map((m, i) => {
            const c = colorMap[m.color];
            const open = !!expanded[i];
            return (
              <div
                key={i}
                style={{
                  border: "0.5px solid var(--color-border-tertiary)",
                  borderRadius: "var(--border-radius-lg)",
                  overflow: "hidden",
                  background: "var(--color-background-primary)",
                }}
              >
                <button
                  onClick={() => toggle(i)}
                  style={{
                    width: "100%", textAlign: "left", background: "none", border: "none",
                    padding: "14px 16px", cursor: "pointer", display: "flex", alignItems: "center", gap: 12,
                  }}
                >
                  <span style={{
                    fontSize: 10, fontWeight: 500, padding: "2px 8px",
                    background: c.tag, color: c.tagText,
                    borderRadius: "var(--border-radius-md)", letterSpacing: "0.06em", whiteSpace: "nowrap"
                  }}>{m.stage}</span>
                  <span style={{ flex: 1, fontSize: 14, fontWeight: 500, color: "var(--color-text-primary)" }}>{m.name}</span>
                  <span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>{open ? "▲" : "▼"}</span>
                </button>
                <div style={{ fontSize: 13, color: "var(--color-text-secondary)", padding: "0 16px 4px" }}>
                  {m.task}
                </div>
                {open && (
                  <div style={{ borderTop: "0.5px solid var(--color-border-tertiary)", padding: 16 }}>
                    {[
                      ["Architecture", m.arch],
                      ["Input", m.input],
                      ["Output", m.output],
                      ["Inference", m.inference],
                    ].map(([label, val]) => (
                      <div key={label} style={{ display: "flex", gap: 12, marginBottom: 10 }}>
                        <span style={{ fontSize: 12, color: "var(--color-text-tertiary)", minWidth: 80, paddingTop: 1 }}>{label}</span>
                        <span style={{ fontSize: 13, color: "var(--color-text-primary)", lineHeight: 1.6 }}>{val}</span>
                      </div>
                    ))}
                    <div style={{
                      background: c.bg, borderRadius: "var(--border-radius-md)",
                      padding: "10px 12px", marginBottom: 10
                    }}>
                      <div style={{ fontSize: 11, color: c.text, fontWeight: 500, marginBottom: 4, letterSpacing: "0.06em" }}>WHY THIS MODEL</div>
                      <div style={{ fontSize: 13, color: c.text, lineHeight: 1.6 }}>{m.why}</div>
                    </div>
                    <div style={{ fontSize: 12, color: "var(--color-text-secondary)", lineHeight: 1.6 }}>
                      <span style={{ color: "var(--color-text-tertiary)" }}>Alternatives: </span>{m.alt}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* DATASETS TAB */}
      {tab === "Datasets" && (
        <div>
          {datasets.map((cat, ci) => {
            const c = colorMap[cat.color];
            return (
              <div key={ci} style={{ marginBottom: 24 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                  <div style={{ height: 1, width: 12, background: c.border }} />
                  <span style={{ fontSize: 11, fontWeight: 500, color: c.text, letterSpacing: "0.1em", textTransform: "uppercase" }}>
                    {cat.category}
                  </span>
                  <div style={{ flex: 1, height: "0.5px", background: "var(--color-border-tertiary)" }} />
                </div>
                {cat.items.map((d, di) => (
                  <div key={di} style={{
                    border: "0.5px solid var(--color-border-tertiary)",
                    borderRadius: "var(--border-radius-lg)",
                    padding: "14px 16px",
                    marginBottom: 8,
                    background: "var(--color-background-primary)",
                  }}>
                    <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 6 }}>
                      <span style={{ fontSize: 14, fontWeight: 500, color: "var(--color-text-primary)" }}>{d.name}</span>
                      <span style={{
                        fontSize: 11, padding: "1px 8px",
                        background: c.tag, color: c.tagText,
                        borderRadius: "var(--border-radius-md)"
                      }}>{d.size}</span>
                      <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginLeft: "auto" }}>{d.license}</span>
                    </div>
                    <div style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 6, lineHeight: 1.6 }}>
                      {d.use}
                    </div>
                    <div style={{
                      fontSize: 12, color: c.text,
                      background: c.bg, borderRadius: "var(--border-radius-md)",
                      padding: "6px 10px", lineHeight: 1.5
                    }}>
                      {d.note}
                    </div>
                    {d.link && (
                      <a href={d.link} style={{ fontSize: 11, color: "var(--color-text-info)", display: "block", marginTop: 8 }}>
                        {d.link}
                      </a>
                    )}
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      )}

      {/* OPEN QUESTIONS TAB */}
      {tab === "Open Questions" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {questions.map((q, i) => (
            <div key={i} style={{
              border: "0.5px solid var(--color-border-tertiary)",
              borderRadius: "var(--border-radius-lg)",
              padding: "14px 16px",
              background: "var(--color-background-primary)",
            }}>
              <div style={{ display: "flex", alignItems: "flex-start", gap: 10, marginBottom: 8 }}>
                <span style={{ fontSize: 14, fontWeight: 500, color: "var(--color-text-primary)", flex: 1, lineHeight: 1.5 }}>
                  {q.q}
                </span>
                <span style={{
                  fontSize: 11, fontWeight: 500, padding: "2px 8px", whiteSpace: "nowrap",
                  background: priorityBg[q.priority], color: priorityColor[q.priority],
                  borderRadius: "var(--border-radius-md)",
                }}>{q.priority}</span>
              </div>
              <div style={{ fontSize: 13, color: "var(--color-text-secondary)", lineHeight: 1.7 }}>
                {q.detail}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
