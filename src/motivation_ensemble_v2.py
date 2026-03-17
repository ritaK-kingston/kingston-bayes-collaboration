#!/usr/bin/env python3
"""
Motivational Content Classification (V2 Ensemble)

This module implements the complete BERT-based ensemble used in the
Kingston–Bayes crowdfunding motivation study.

It includes:
  - Keyword dictionaries and specificity weights
  - Preprocessing pipeline for fundraiser narratives
  - BART-Large-MNLI zero-shot classifier
  - DistilRoBERTa emotion model
  - Twitter-RoBERTa sentiment model
  - Ensemble weighting and confidence-squared profiling

The main entry-point is `run_analysis`, which takes a CSV of fundraiser
stories and produces per-story motivation profiles and category labels.
"""

import os
import sys
import re
import json
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd
import numpy as np

try:
    from transformers import pipeline
    import torch
    DEVICE = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Transformers loaded. Device: {DEVICE}")
except ImportError:
    print("Installing transformers + torch ...")
    os.system("pip install transformers torch")
    from transformers import pipeline
    import torch
    DEVICE = "cpu"


# ── 1. Keyword dictionaries ─────────────────────────────────────────────

REFINED_MOTIVATION_KEYWORDS: Dict[str, List[str]] = {
    "Advocacy": [
        "raise awareness for",
        "campaign to support",
        "advocate for change",
        "speak out against",
        "fight for justice",
        "stand up for",
        "policy reform",
        "equality rights",
        "social justice",
        "activism",
        "protest",
        "petition",
        "voice concerns",
        "make a difference",
        "social movement",
    ],
    "Altruism and Empathy": [
        "save lives",
        "caring for others",
        "kindness",
        "support those in need",
        "change lives",
        "bring hope",
        "selfless giving",
        "give hope",
        "anonymous donation",
        "concern for others",
        "pure giving",
        "act of kindness",
        "help others",
        "relieve suffering",
        "compassionate giving",
    ],
    "Close to Home": [
        "local community",
        "our neighborhood",
        "nearby school",
        "regional hospital",
        "area development",
        "community project",
        "local charity",
        "neighborhood initiative",
        "regional support",
        "nearby cause",
        "local school",
        "community initiative",
        "area support",
        "local development",
        "community cause",
    ],
    "Stewardship": [
        "donation management",
        "fund allocation",
        "charity oversight",
        "financial stewardship",
        "resource management",
        "ensure funds are used",
        "directly support",
        "transparent giving",
        "responsible management",
        "efficient allocation",
        "accountable use",
        "fund accountability",
        "charity administration",
        "transparent donation",
        "financial oversight",
    ],
    "Seeking Experiences": [
        "challenge myself",
        "personal challenge",
        "charity event",
        "fundraising challenge",
        "sponsored activity",
        "charity run",
        "challenge event",
        "fundraising event",
        "marathon training",
        "sponsored walk",
        "exciting challenge",
        "personal achievement",
        "adventure",
        "thrilling experience",
        "participate in",
    ],
    "Moral Obligation": [
        "justice",
        "compassion",
        "humanity",
        "help others",
        "moral duty",
        "ethical giving",
        "compassionate",
        "commitment to help",
        "duty to support",
        "moral responsibility",
        "ethical obligation",
        "human rights",
        "social responsibility",
        "moral imperative",
        "ethical commitment",
    ],
    "Close to the Heart": [
        "in memory of",
        "close to my heart",
        "personal connection",
        "family member",
        "loved one",
        "because of my experience",
        "family friend",
        "personal loss",
        "memory tribute",
        "dedicated to",
        "personal experience",
        "close family",
        "personal story",
        "family support",
        "personal dedication",
    ],
    "Personal Development": [
        "personal growth",
        "learn new skills",
        "develop myself",
        "overcome challenges",
        "achieve goals",
        "personal journey",
        "skill development",
        "learning journey",
        "self-improvement",
        "capability building",
        "personal achievement",
        "growth opportunity",
        "develop skills",
        "personal transformation",
        "skill building",
    ],
    "Social Standing": [
        "like and share",
        "friends support",
        "social media",
        "post about",
        "brand recognition",
        "support me",
        "social recognition",
        "identity expression",
        "social influence",
        "networking",
        "social status",
        "public image",
        "social visibility",
        "social networking",
        "social presence",
    ],
}

CATEGORIES: List[str] = list(REFINED_MOTIVATION_KEYWORDS.keys())


SPECIFICITY_WEIGHTS: Dict[str, float] = {
    # Highly specific phrases (weight 1.0)
    "marathon training": 1.0,
    "charity run": 1.0,
    "fundraising event": 1.0,
    "sponsored walk": 1.0,
    "donation management": 1.0,
    "fund allocation": 1.0,
    "charity oversight": 1.0,
    "financial stewardship": 1.0,
    "local community": 1.0,
    "our neighborhood": 1.0,
    "nearby school": 1.0,
    "regional hospital": 1.0,
    "raise awareness for": 1.0,
    "campaign to support": 1.0,
    "advocate for change": 1.0,
    "in memory of": 1.0,
    "close to my heart": 1.0,
    "personal connection": 1.0,
    "challenge myself": 1.0,
    "personal challenge": 1.0,
    "charity event": 1.0,
    "personal growth": 1.0,
    "learn new skills": 1.0,
    "develop myself": 1.0,
    "like and share": 1.0,
    "friends support": 1.0,
    "social media": 1.0,
    # Moderately specific (0.7)
    "community project": 0.7,
    "local charity": 0.7,
    "neighborhood initiative": 0.7,
    "transparent giving": 0.7,
    "responsible management": 0.7,
    "efficient allocation": 0.7,
    "policy reform": 0.7,
    "equality rights": 0.7,
    "social justice": 0.7,
    "family member": 0.7,
    "loved one": 0.7,
    "personal experience": 0.7,
    "fundraising challenge": 0.7,
    "sponsored activity": 0.7,
    "exciting challenge": 0.7,
    "skill development": 0.7,
    "learning journey": 0.7,
    "self-improvement": 0.7,
    "social recognition": 0.7,
    "identity expression": 0.7,
    "social influence": 0.7,
    "caring for others": 0.7,
    "support those in need": 0.7,
    "change lives": 0.7,
    "justice": 0.7,
    "compassion": 0.7,
    "humanity": 0.7,
    # Generic terms (≤0.5) often ignored by MIN_KEYWORD_WEIGHT
    "help others": 0.3,
    "support": 0.3,
    "help": 0.3,
    "community": 0.3,
    "local": 0.3,
    "challenge": 0.3,
    "event": 0.3,
    "experience": 0.3,
    "growth": 0.3,
    "development": 0.3,
    "family": 0.3,
    "friend": 0.3,
    "personal": 0.3,
    "social": 0.3,
    "fun": 0.1,
    "giving": 0.1,
    "use": 0.1,
    "like": 0.1,
    "share": 0.1,
    "post": 0.1,
    "follow": 0.1,
}


# ── 2. Configuration (weights, thresholds) ──────────────────────────────

W_ZERO_SHOT = 0.65
W_KEYWORDS = 0.15
W_EMOTION = 0.15
W_SENTIMENT = 0.05

ZERO_SHOT_TOP_K = 5

TEMPERATURE = 0.7
MIN_KEYWORD_WEIGHT = 0.5
MULTI_LABEL_THRESHOLD = 0.55
MAX_CATEGORIES_PER_STORY = 5
CONFIDENCE_THRESHOLD = 0.55
BELOW_THRESHOLD_FACTOR = 0.3


EMOTION_TO_MOTIVATION: Dict[str, List[str]] = {
    "joy": ["Personal Development", "Seeking Experiences"],
    "sadness": ["Close to the Heart", "Altruism and Empathy"],
    "anger": ["Advocacy", "Moral Obligation"],
    "fear": ["Close to the Heart", "Stewardship"],
    "surprise": ["Seeking Experiences", "Social Standing"],
    "disgust": ["Advocacy", "Moral Obligation"],
    "neutral": ["Close to Home", "Stewardship"],
}


SENTIMENT_TO_MOTIVATION: Dict[str, List[str]] = {
    "positive": ["Personal Development", "Seeking Experiences", "Social Standing"],
    "negative": ["Close to the Heart", "Altruism and Empathy", "Moral Obligation"],
    "neutral": ["Stewardship", "Close to Home", "Advocacy"],
}


# ── 3. Preprocessing ────────────────────────────────────────────────────

def preprocess_text(text: Any) -> str:
    """HTML/URL removal, whitespace normalisation, truncation."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = re.sub(r"<[^>]+>", "", str(text))
    text = re.sub(r"http[s]?://\\S+", "", text)
    text = re.sub(r"\\s+", " ", text).strip()
    if len(text) > 2500:
        text = text[:2500]
    return text


def compile_keyword_patterns(
    motivation_keywords: Dict[str, List[str]]
) -> Dict[str, List[Any]]:
    """Pre-compile keyword regexes with word-boundary constraints."""
    patterns: Dict[str, List[Any]] = {}
    for category, keywords in motivation_keywords.items():
        compiled = []
        for kw in keywords:
            escaped = re.escape(kw).replace(r"\\ ", r"\\s+")
            pattern = rf"(?i)(?<!\\w){escaped}(?!\\w)"
            compiled.append((kw, re.compile(pattern)))
        patterns[category] = compiled
    return patterns


def calculate_keyword_scores(
    text: str, keyword_patterns: Dict[str, List[Any]]
) -> Dict[str, float]:
    """Compute raw keyword scores per category with specificity weights."""
    if not text:
        return {cat: 0.0 for cat in CATEGORIES}
    scores: Dict[str, float] = {}
    for category, patterns in keyword_patterns.items():
        score = 0.0
        for kw, pat in patterns:
            weight = SPECIFICITY_WEIGHTS.get(kw, 0.5)
            if weight < MIN_KEYWORD_WEIGHT:
                continue
            matches = len(pat.findall(text))
            if matches:
                score += weight * min(matches, 3)
        scores[category] = score
    return scores


# ── 4. Model loading ────────────────────────────────────────────────────

def load_models() -> Dict[str, Any]:
    """Load zero-shot, emotion, and sentiment models."""
    print("Loading models ...")
    models: Dict[str, Any] = {}
    try:
        models["emotion"] = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
            device=DEVICE,
        )
        print("  Emotion classifier loaded")
    except Exception as e:
        print(f"  Emotion classifier FAILED: {e}")
        models["emotion"] = None

    try:
        models["sentiment"] = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            return_all_scores=True,
            device=DEVICE,
        )
        print("  Sentiment classifier loaded")
    except Exception as e:
        print(f"  Sentiment classifier FAILED: {e}")
        models["sentiment"] = None

    try:
        models["zero_shot"] = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=DEVICE,
        )
        print("  Zero-shot classifier loaded")
    except Exception as e:
        print(f"  Zero-shot classifier FAILED: {e}")
        models["zero_shot"] = None

    return models


# ── 5. Component analysis ───────────────────────────────────────────────

def analyze_emotions(text: str, clf: Any) -> Dict[str, float]:
    if not clf or not text or len(text.strip()) < 10:
        return {}
    try:
        results = clf(text)
        if isinstance(results, list) and results and isinstance(results[0], list):
            results = results[0]
        return {r["label"]: r["score"] for r in results if isinstance(r, dict)}
    except Exception:
        return {}


def analyze_sentiment(text: str, clf: Any) -> Dict[str, float]:
    if not clf or not text or len(text.strip()) < 10:
        return {}
    try:
        results = clf(text)
        if isinstance(results, list) and results and isinstance(results[0], list):
            results = results[0]
        return {r["label"]: r["score"] for r in results if isinstance(r, dict)}
    except Exception:
        return {}


def analyze_zero_shot(text: str, clf: Any) -> Dict[str, float]:
    """Zero-shot classification with top-K filtering."""
    if not clf or not text or len(text.strip()) < 10:
        return {}
    try:
        result = clf(text, CATEGORIES)
        all_scores = dict(zip(result["labels"], result["scores"]))
        top_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[
            :ZERO_SHOT_TOP_K
        ]
        top_sum = sum(s for _, s in top_items) or 1.0
        return {lab: sc / top_sum for lab, sc in top_items}
    except Exception:
        return {}


# ── 6. Ensemble scoring and multi-label assignment ──────────────────────

def compute_ensemble_scores(
    zero_shot_scores: Dict[str, float],
    keyword_scores: Dict[str, float],
    emotion_scores: Dict[str, float],
    sentiment_scores: Dict[str, float],
) -> Dict[str, float]:
    """Combine all components into per-category scores."""
    category_scores: Dict[str, float] = {}

    for category in CATEGORIES:
        score = 0.0

        score += zero_shot_scores.get(category, 0.0) * W_ZERO_SHOT

        kw_raw = keyword_scores.get(category, 0.0)
        score += (1.0 - np.exp(-kw_raw / 5.0)) * W_KEYWORDS

        if emotion_scores:
            e_sum = 0.0
            for emotion, emotion_score in emotion_scores.items():
                if (
                    emotion in EMOTION_TO_MOTIVATION
                    and category in EMOTION_TO_MOTIVATION[emotion]
                ):
                    e_sum += emotion_score
            score += e_sum * W_EMOTION

        if sentiment_scores:
            s_sum = 0.0
            for sentiment, sentiment_score in sentiment_scores.items():
                if (
                    sentiment in SENTIMENT_TO_MOTIVATION
                    and category in SENTIMENT_TO_MOTIVATION[sentiment]
                ):
                    s_sum += sentiment_score
            score += s_sum * W_SENTIMENT

        category_scores[category] = score

    return category_scores


def apply_multilabel_classification(
    category_scores: Dict[str, float]
) -> Dict[str, Any]:
    """Apply sigmoid, threshold, and multi-label selection."""
    cats = list(category_scores.keys())
    vals = np.array([category_scores[c] for c in cats], dtype=float)

    scaled = vals / max(TEMPERATURE, 1e-6)
    probs = 1.0 / (1.0 + np.exp(-scaled))

    above = [(c, p) for c, p in zip(cats, probs) if p >= MULTI_LABEL_THRESHOLD]
    above.sort(key=lambda x: x[1], reverse=True)
    selected = above[:MAX_CATEGORIES_PER_STORY]

    all_probs = dict(zip(cats, probs.tolist()))

    if selected:
        primary = selected[0][0]
        primary_conf = selected[0][1]
    else:
        primary = "No Category"
        primary_conf = 0.0

    return {
        "primary_category": primary,
        "primary_confidence": primary_conf,
        "all_categories": [c for c, _ in selected],
        "num_categories": len(selected),
        "all_probs": all_probs,
        "all_raw_scores": category_scores,
    }


def confidence_squared_profile(all_probs: Dict[str, float]) -> Dict[str, float]:
    """
    Compute normalised motivation profile using confidence-squared weighting.

    For each category probability p:
      - if p >= CONFIDENCE_THRESHOLD → contribution = p^2
      - else                        → contribution = 0.3 * p^2
    The contributions are then normalised to sum to 1.0.
    """
    weighted: Dict[str, float] = {}
    for cat, p in all_probs.items():
        if p >= CONFIDENCE_THRESHOLD:
            weighted[cat] = p ** 2
        else:
            weighted[cat] = (p ** 2) * BELOW_THRESHOLD_FACTOR
    total = sum(weighted.values()) or 1.0
    return {cat: w / total for cat, w in weighted.items()}


# ── 7. Main pipeline ────────────────────────────────────────────────────

def run_analysis(
    csv_path: str,
    story_col: str = "story",
    id_col: str = "short_name",
    sample_size: int = None,
    output_dir: str = "results",
) -> pd.DataFrame:
    """
    Run the full V2 ensemble on a CSV of fundraiser narratives.

    Required columns:
      - `story_col` (default: 'story') – the narrative text
    Optional:
      - `id_col` (default: 'short_name') – unique identifier per story
      - `activity_type` – campaign type (if present, copied through)
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    if story_col not in df.columns:
        raise ValueError(
            f"Column '{story_col}' not found. Available: {list(df.columns)}"
        )
    if id_col not in df.columns:
        df[id_col] = df.index.astype(str)

    print("Preprocessing ...")
    df["clean_story"] = df[story_col].apply(preprocess_text)
    df = df[df["clean_story"].str.len() >= 10].copy()

    norm = (
        df["clean_story"]
        .str.lower()
        .str.replace(r"\\s+", " ", regex=True)
        .str.strip()
    )
    before = len(df)
    df = df.loc[~norm.duplicated()].copy()
    print(f"Deduplicated: {before} -> {len(df)} unique stories")

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled {sample_size} stories")

    models = load_models()
    keyword_patterns = compile_keyword_patterns(REFINED_MOTIVATION_KEYWORDS)

    checkpoint_path = os.path.join(output_dir, "checkpoint_partial.csv")
    start_idx = 0
    results: List[Dict[str, Any]] = []
    if os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path)
        results = existing.to_dict("records")
        start_idx = len(results)
        print(f"Resuming from checkpoint: {start_idx} stories already done")

    total = len(df)
    print(f"Analysing {total} stories (starting from {start_idx}) ...")
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx < start_idx:
            continue
        if idx % 50 == 0:
            print(f"  {idx}/{total} ...")
        text = row["clean_story"]

        zs = analyze_zero_shot(text, models["zero_shot"])
        kw = calculate_keyword_scores(text, keyword_patterns)
        em = analyze_emotions(text, models["emotion"])
        se = analyze_sentiment(text, models["sentiment"])

        raw_scores = compute_ensemble_scores(zs, kw, em, se)
        ml = apply_multilabel_classification(raw_scores)
        profile = confidence_squared_profile(ml["all_probs"])

        results.append(
            {
                id_col: row[id_col],
                "clean_story": text,
                "story_length": len(text.split()),
                "activity_type": row.get("activity_type", ""),
                "zero_shot_scores": json.dumps(zs),
                "keyword_scores": json.dumps(kw),
                "emotion_scores": json.dumps(em),
                "sentiment_scores": json.dumps(se),
                "all_raw_scores": json.dumps(raw_scores),
                "all_probs": json.dumps(ml["all_probs"]),
                "primary_category": ml["primary_category"],
                "primary_confidence": ml["primary_confidence"],
                "all_categories": json.dumps(ml["all_categories"]),
                "num_categories": ml["num_categories"],
                "motivation_profile": json.dumps(profile),
                **{f"profile_{cat}": profile.get(cat, 0.0) for cat in CATEGORIES},
            }
        )

        if (idx + 1) % 500 == 0:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            print(f"  Checkpoint saved at {idx + 1}/{total}")

    results_df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, f"motivation_ensemble_v2_results_{timestamp}.csv")
    results_df.to_csv(out_path, index=False)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print(f"Results saved to {out_path}")

    return results_df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.motivation_ensemble_v2 <input.csv> [sample_size] [output_dir]")
        sys.exit(1)
    csv_file = sys.argv[1]
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else None
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "results"
    run_analysis(csv_file, sample_size=sample, output_dir=out_dir)

