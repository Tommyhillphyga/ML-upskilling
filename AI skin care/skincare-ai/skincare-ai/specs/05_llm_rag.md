# 05 — LLM / RAG Layer

The LLM layer handles everything language-facing: skin reports, routine explanations, ingredient rationale, progress narratives, and in-app Q&A. It sits at Stage 07 and uses a RAG (retrieval-augmented generation) architecture backed by a dermatology knowledge base.

No LLM is trained or fine-tuned from scratch. You use a capable base model with structured prompting and retrieval.

---

## Architecture

```
Structured CV output (SeverityReport JSON)
            │
            ▼
    [RAG Retrieval Layer]
    ┌─────────────────────────────┐
    │  Query 1: Ingredient rules  │ → Pinecone / pgvector
    │  Query 2: Routine templates │ → dermatologist YAML
    │  Query 3: Ingredient safety │ → Open Beauty Facts index
    └─────────────────────────────┘
            │
            ▼
    [Context Assembly]
    Concern list + severity + retrieved docs + user profile
            │
            ▼
    [LLM — Claude claude-sonnet-4-20250514]
    System prompt + assembled context + task instruction
            │
            ▼
    Structured JSON response (RoutineResponse)
```

---

## Model selection

**Primary:** Claude claude-sonnet-4-20250514 (Anthropic API)
**Fallback:** GPT-4o (OpenAI API)

Claude is preferred for instruction-following and safe output on clinical-adjacent content. Switch to GPT-4o if rate limits or cost become an issue at scale.

**Do not use:** Llama 3 or other self-hosted open-weights models for the first production version. Clinical accuracy on ingredient interactions is too high-stakes for an unvalidated open model without significant red-teaming.

---

## Knowledge base

### Structure
```
data/knowledge_base/
├── clinical/
│   ├── concern_routines.yaml          # Dermatologist-authored routine templates
│   ├── ingredient_conflicts.yaml      # Conflict + resolution rules
│   ├── ingredient_glossary.md         # Plain-English explanations per ingredient
│   └── concern_explanations.md        # What each concern means, plain language
├── products/
│   ├── ingredient_index.json          # Ingredient → product list (from Open Beauty Facts)
│   └── ingredient_function_map.json   # Ingredient → function labels
└── regulatory/
    └── pregnancy_unsafe.yaml          # Ingredients to flag for pregnant users
```

### Embedding and indexing
```python
from openai import OpenAI
import pinecone
from pathlib import Path
import yaml
import json

def embed_text(text: str, model="text-embedding-3-small") -> list:
    client = OpenAI()
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def index_knowledge_base(kb_dir: Path, index_name: str):
    pc = pinecone.Pinecone()
    index = pc.Index(index_name)

    documents = []

    # Index ingredient glossary
    glossary_path = kb_dir / "clinical/ingredient_glossary.md"
    glossary_text = glossary_path.read_text()
    # Split by ingredient section (## Ingredient Name)
    sections = glossary_text.split("\n## ")[1:]
    for section in sections:
        ingredient_name = section.split("\n")[0].strip()
        documents.append({
            "id": f"ingredient_{ingredient_name.replace(' ', '_').lower()}",
            "text": section,
            "metadata": {"type": "ingredient", "name": ingredient_name}
        })

    # Index routine templates
    routines = yaml.safe_load((kb_dir / "clinical/concern_routines.yaml").read_text())
    for concern, levels in routines.items():
        for severity, routine in levels.items():
            text = yaml.dump({concern: {severity: routine}})
            documents.append({
                "id": f"routine_{concern}_{severity}",
                "text": text,
                "metadata": {"type": "routine", "concern": concern, "severity": severity}
            })

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        vectors = [(d["id"], embed_text(d["text"]), d["metadata"]) for d in batch]
        index.upsert(vectors=vectors)
```

---

## Prompt templates

### 1. Skin report generation

```python
SKIN_REPORT_SYSTEM_PROMPT = """
You are a skincare assistant working alongside licensed dermatologists.
Your role is to explain a user's skin scan results in clear, supportive, non-alarmist language.

Rules:
- Never use clinical diagnostic language ("you have rosacea", "you have acne vulgaris").
  Use descriptive language ("your skin shows signs of...", "we detected...")
- Always normalise concerns — most people have at least 2-3 skin concerns. Never make the user feel bad.
- Be specific about location (chin, forehead, cheeks) using the zone data provided.
- Keep explanations under 3 sentences per concern.
- Do not recommend specific products by brand name. Focus on ingredients.
- If severity is high on any concern, gently suggest a dermatologist consultation.
- Output must be valid JSON matching the SkinReportResponse schema.
"""

def build_skin_report_prompt(severity_report: dict, zone_map: dict) -> str:
    return f"""
Generate a skin report from the following scan results.

Detected concerns and severity (0-100 scale, higher = more severe):
{json.dumps(severity_report['severity_scores'], indent=2)}

Zone mapping (which concern appears where on the face):
{json.dumps(zone_map, indent=2)}

User's Fitzpatrick skin type: {severity_report.get('fitzpatrick', 'unknown')}

Output a JSON object with this structure:
{{
  "headline": "One encouraging sentence summarising the overall scan",
  "overall_skin_score": <number 0-100>,
  "concerns": [
    {{
      "concern": "acne",
      "headline": "Short friendly label (e.g. 'Some breakout activity')",
      "explanation": "2-3 sentences explaining what was detected and where",
      "severity_label": "Mild | Moderate | Significant | Severe",
      "zones": ["chin", "t_zone"]
    }}
  ]
}}
"""
```

### Example skin report output
```json
{
  "headline": "Your skin is looking mostly clear with a few areas to focus on.",
  "overall_skin_score": 68,
  "concerns": [
    {
      "concern": "acne",
      "headline": "Some breakout activity",
      "explanation": "We detected moderate breakout activity concentrated around your chin and T-zone — a common pattern often linked to hormonal changes or sebum production. The good news is this responds well to the right routine.",
      "severity_label": "Moderate",
      "zones": ["chin", "t_zone"]
    },
    {
      "concern": "dark_circles",
      "headline": "Under-eye pigmentation",
      "explanation": "There's some noticeable pigmentation in the under-eye area. This is extremely common and often has both genetic and lifestyle components. Targeted ingredients can make a visible difference over time.",
      "severity_label": "Significant",
      "zones": ["eye_area"]
    }
  ]
}
```

---

### 2. Routine generation

```python
ROUTINE_SYSTEM_PROMPT = """
You are a skincare formulation assistant. You generate personalised morning and night skincare routines
based on detected skin concerns, dermatologist-validated templates, and the user's selected products.

Rules:
- Maximum 4 steps per routine (cleanser, treatment, moisturiser, SPF for AM; cleanser, treatment, moisturiser for PM)
- Beginner-first: assume the user has never had a skincare routine before
- Explain each step in 1-2 sentences, including WHY it helps this person specifically
- Flag any ingredient conflicts with a clear caution note
- If the user has selected products, incorporate them. If not, describe what to look for.
- Never recommend more than 1 active ingredient per session (no stacking actives)
- Output valid JSON matching the RoutineResponse schema
"""

def build_routine_prompt(
    ingredient_list: dict,
    user_products: list,
    concern_summary: dict,
    conflict_rules: list,
    routine_templates: dict
) -> str:
    return f"""
Generate a morning and night routine for this user.

Recommended active ingredients (prioritised):
{json.dumps(ingredient_list, indent=2)}

User's existing products at home:
{json.dumps(user_products, indent=2)}

Skin concerns summary:
{json.dumps(concern_summary, indent=2)}

Known ingredient conflicts to check:
{json.dumps(conflict_rules, indent=2)}

Dermatologist routine templates for reference:
{json.dumps(routine_templates, indent=2)}

Output a RoutineResponse JSON object.
"""
```

### Example routine output
```json
{
  "routine_id": "routine_20240318_u1042",
  "morning": [
    {
      "step": 1,
      "step_name": "Cleanse",
      "product_type": "Gentle foaming cleanser",
      "key_ingredient": "salicylic_acid",
      "instruction": "Wet your face with lukewarm water, apply a small amount, and gently massage for 60 seconds. Rinse thoroughly. Using salicylic acid in your cleanser helps clear congestion in your pores without over-drying.",
      "user_product": null,
      "what_to_look_for": "Look for '1-2% salicylic acid' on the label. BHA or beta-hydroxy acid means the same thing."
    },
    {
      "step": 2,
      "step_name": "Moisturise",
      "product_type": "Lightweight gel moisturiser",
      "key_ingredient": "niacinamide",
      "instruction": "Apply a pea-sized amount all over your face. Niacinamide helps regulate the oil production on your T-zone and chin while keeping your cheeks hydrated.",
      "user_product": "CeraVe PM Facial Moisturising Lotion",
      "what_to_look_for": null
    },
    {
      "step": 3,
      "step_name": "SPF",
      "product_type": "Mineral sunscreen SPF 30+",
      "key_ingredient": "zinc_oxide",
      "instruction": "Apply generously as the last step every morning. This is non-negotiable — UV exposure worsens every skin concern you have, especially hyperpigmentation and dark circles.",
      "user_product": null,
      "what_to_look_for": "Look for SPF 30 or higher with zinc oxide. Mineral formulas are less likely to irritate sensitive skin."
    }
  ],
  "night": [
    {
      "step": 1,
      "step_name": "Cleanse",
      "product_type": "Gentle cleanser",
      "key_ingredient": null,
      "instruction": "Remove SPF and any product buildup before your treatment steps. Doesn't need to be medicated in the evening.",
      "user_product": null,
      "what_to_look_for": "Any gentle, pH-balanced cleanser without fragrance."
    },
    {
      "step": 2,
      "step_name": "Treatment",
      "product_type": "Retinol eye cream",
      "key_ingredient": "retinol",
      "instruction": "Apply a small amount under your eyes only. Retinol builds collagen over time and helps with the dark pigmentation we detected. Start just 2 nights per week to let your skin adjust.",
      "user_product": null,
      "what_to_look_for": "Look for 0.025-0.05% retinol for beginners. Anything stronger will cause too much irritation.",
      "caution": "Do not use retinol if pregnant. Introduce slowly — redness and flaking in the first 2 weeks is normal. If it persists past week 4, stop and consult a dermatologist."
    },
    {
      "step": 3,
      "step_name": "Moisturise",
      "product_type": "Barrier repair moisturiser",
      "key_ingredient": "ceramides",
      "instruction": "Apply generously over the whole face. On nights you use retinol, layer this on top to buffer any irritation.",
      "user_product": "CeraVe PM Facial Moisturising Lotion",
      "what_to_look_for": null
    }
  ],
  "conflict_warnings": [],
  "notes": "Introduce one new product at a time. Wait 2 weeks before adding anything new so you can identify what your skin is reacting to."
}
```

---

### 3. Progress narrative

```python
PROGRESS_SYSTEM_PROMPT = """
You generate encouraging, honest progress reports comparing a user's current skin scan to their baseline.
Be specific and cite numbers. If things have worsened, acknowledge it honestly without alarm.
Never overstate results. The user is comparing scan data, not buying a product.
"""

PROGRESS_PROMPT_TEMPLATE = """
The user completed a follow-up scan after {weeks} weeks on their routine.

Progress data (positive delta = improvement):
{progress_json}

Routine they were following:
{routine_summary}

Write a progress report (3-5 sentences) that:
1. Opens with the headline metric (biggest improvement or most concerning regression)
2. Explains the likely cause (routine adherence, ingredient efficacy, or external factors)
3. Recommends whether to continue, adjust, or escalate to a dermatologist
Output as JSON: {{ "headline": "...", "body": "...", "recommendation": "continue | adjust | consult_derm" }}
"""
```

---

## RAG retrieval function

```python
def retrieve_relevant_context(
    concerns: list,
    severity: dict,
    user_products: list,
    top_k: int = 5
) -> dict:
    """Retrieve relevant documents from the knowledge base for the given concerns."""

    query_parts = [
        f"routine for {c} skin concern severity {severity.get(c, {}).get('label', 'moderate')}"
        for c in concerns
    ]
    query_parts += [f"ingredient safety and use for {p}" for p in user_products[:3]]

    all_results = {}
    for query in query_parts:
        query_embedding = embed_text(query)
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        for match in results.matches:
            doc_id = match.id
            if doc_id not in all_results or match.score > all_results[doc_id]['score']:
                all_results[doc_id] = {
                    'score': match.score,
                    'text': match.metadata.get('text', ''),
                    'type': match.metadata.get('type', '')
                }

    # Sort by score, return top results
    sorted_docs = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)
    return {
        'routine_templates': [d for d in sorted_docs if d['type'] == 'routine'][:3],
        'ingredient_info': [d for d in sorted_docs if d['type'] == 'ingredient'][:5],
    }
```

---

## LLM call wrapper

```python
import anthropic
import json

client = anthropic.Anthropic()

def call_llm(system_prompt: str, user_prompt: str,
             max_tokens: int = 2000, temperature: float = 0.3) -> dict:
    """
    temperature=0.3: Low for clinical content — reduces hallucination risk.
    Always parse output as JSON. Validate against schema before returning.
    """
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    raw_text = message.content[0].text

    try:
        # Strip markdown code blocks if present
        clean = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw_text[:500]}")
```

---

## Evaluation

Every prompt template must be evaluated before going to users. The dermatologist reviews 50 sample outputs per template and scores:

| Criterion | Scale | Pass threshold |
|---|---|---|
| Clinical accuracy | 1–5 | ≥4.0 |
| Tone (not alarmist, not dismissive) | 1–5 | ≥4.0 |
| Actionability (user knows what to do) | 1–5 | ≥4.0 |
| No false claims or overstatements | Pass/Fail | 100% Pass |
| JSON schema validity | Pass/Fail | 100% Pass |

See [08_evaluation.md](./08_evaluation.md) for the full LLM evaluation rubric.
