## Kingston–Bayes Crowdfunding Motivation & Donor Analytics

This repository packages the **core, reusable components** from the Kingston–Bayes
crowdfunding motivation study, so that collaborators can run the classification
pipeline independently of the original data-collection environment.

It focuses on four elements:

- **Keyword dictionaries** used for the motivation categories
- **BART-Large-MNLI zero-shot classifier** configuration
- **Emotion and sentiment models** used as ensemble components
- **Preprocessing, ensemble weighting, and final category assignment logic**

All model code here is **data-agnostic** – you supply your own CSV of fundraiser
stories (or similar narratives), and the pipeline will produce multi-label
motivation classifications and confidence-weighted profiles.

---

### 1. Repository Structure

- `src/motivation_ensemble_v2.py`  
  Complete implementation of the **V2 BERT-based ensemble**, including:
  - Keyword dictionaries and specificity weights
  - Text preprocessing
  - Model loading (zero-shot, emotion, sentiment)
  - Ensemble scoring and thresholding
  - Confidence-squared weighting and profile generation
  - Checkpointing and CSV output

You can treat this module as the **single source of truth** for the methodology.

---

### 2. Installation

Create and activate a Python environment (Python 3.9+ recommended), then:

```bash
cd /path/to/kingston-bayes-collaboration

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install pandas numpy scipy matplotlib seaborn scikit-learn transformers torch
```

These are the main libraries used in the original study:

- `transformers`, `torch` – Hugging Face models (BART, DistilRoBERTa, Twitter-RoBERTa)
- `pandas`, `numpy` – data handling and numeric computation

You do **not** need a GPU, but for large datasets a GPU or Apple Silicon (MPS)
will substantially reduce runtime.

---

### 3. Input Data Format

The classifier expects a CSV file with at least:

- **`story`** – the narrative text for each fundraising page (or other story)

Optional but recommended:

- **`short_name`** – unique identifier for each story (e.g. JustGiving `short_name`); if missing, an index-based ID is generated
- **`activity_type`** – campaign type label (carried through into the output)

Example:

```csv
short_name,story,activity_type
RubaTrek,"Assalamu Alaykum and welcome to Aston University Islamic Society's JustGiving page...",CharityAppeal
AlisonandLil,"Team tangerine walking the length of Britain, John O'groats to Lands End...",OtherPersonalChallenge
```

---

### 4. Running the Classifier

From the repository root:

```bash
cd /path/to/kingston-bayes-collaboration
mkdir -p results

python -m src.motivation_ensemble_v2 input_stories.csv
```

Optional arguments:

```bash
python -m src.motivation_ensemble_v2 input_stories.csv 500 results
```

- `500` – sample size (take a random sample of 500 stories for a quick run)
- `results` – output directory (default: `results`)

The script will:

1. **Preprocess** text (HTML + URL removal, whitespace normalization, truncation to 2,500 chars)
2. **Deduplicate** narratives (case-insensitive, whitespace-normalised text)
3. **Load models**:
   - Zero-shot: `facebook/bart-large-mnli`
   - Emotion: `j-hartmann/emotion-english-distilroberta-base`
   - Sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest`
4. For each story:
   - Run zero-shot classification over the 9 motivation categories
   - Run keyword matching and scoring
   - Run emotion and sentiment classifiers
   - Combine all four components into category scores
   - Apply sigmoid + threshold + multi-label selection
   - Compute the confidence-squared motivation profile
5. Write a CSV of per-story results to `results/motivation_ensemble_v2_results_<timestamp>.csv`

Checkpointing:

- Every 500 stories, a partial results file `checkpoint_partial.csv` is saved.
- If the process is interrupted, rerunning the script will **resume** from the
  last checkpoint.

---

### 5. Models and Weighting (Technical Detail)

#### 5.1 Motivation Categories

The ensemble works with **nine motivation categories**:

- Close to Home  
- Close to the Heart  
- Altruism and Empathy  
- Moral Obligation  
- Social Standing  
- Personal Development  
- Seeking Experiences  
- Stewardship  
- Advocacy  

These appear in `CATEGORIES` inside `src/motivation_ensemble_v2.py`.

#### 5.2 Keyword Dictionaries

The `REFINED_MOTIVATION_KEYWORDS` dictionary encodes ~15 curated phrases per
category (post-GPT-4 generation and manual pruning). Example:

- **Close to the Heart**: `"in memory of"`, `"close to my heart"`, `"personal connection"`, `"family member"`, `"loved one"`, `"personal loss"`, etc.
- **Seeking Experiences**: `"personal challenge"`, `"charity event"`, `"marathon training"`, `"sponsored walk"`, etc.

Specificity weighting is handled by the `SPECIFICITY_WEIGHTS` mapping:

- Highly specific phrases: weight = 1.0  
- Moderately specific phrases: weight = 0.7  
- Generic terms (e.g. `help`, `community`, `challenge`): weight ≤ 0.5

Only keywords with weight ≥ `MIN_KEYWORD_WEIGHT` (0.5) are used, to avoid
generic matches dominating the signal.

#### 5.3 Preprocessing

Preprocessing (`preprocess_text`) performs:

- HTML tag removal
- URL removal
- Whitespace normalisation
- Truncation to 2,500 characters

This ensures compatibility with transformer context limits and removes web noise.

#### 5.4 Component Models

The ensemble integrates four components (V2 configuration):

- **Zero-shot classification (65% weight)**  
  - Model: `facebook/bart-large-mnli`  
  - Task: zero-shot classification over the 9 motivation labels  
  - Only the **top 5** labels are retained (`ZERO_SHOT_TOP_K = 5`), then renormalised.

- **Keyword matching (15% weight)**  
  - Uses the refined keyword dictionaries with specificity weights  
  - Raw scores are transformed via **exponential saturation**:  
    \( s_{\text{kw}} = 1 - e^{-\text{raw} / 5} \)

- **Emotion analysis (15% weight)**  
  - Model: `j-hartmann/emotion-english-distilroberta-base`  
  - Output labels: `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`  
  - These are mapped to motivation categories via `EMOTION_TO_MOTIVATION` (e.g.,
    `sadness` → Close to the Heart + Altruism and Empathy).

- **Sentiment analysis (5% weight)**  
  - Model: `cardiffnlp/twitter-roberta-base-sentiment-latest`  
  - Labels: `positive`, `negative`, `neutral`  
  - Mapped to motivation categories via `SENTIMENT_TO_MOTIVATION`.

Each component produces per-category scores. The final **ensemble score** per
category is:

- Zero-shot contribution (65%)  
- Keyword saturation contribution (15%)  
- Emotion contribution (15%)  
- Sentiment contribution (5%)

No category priors are used in V2 (all implicit priors = 1.0).

#### 5.5 From Scores to Probabilities and Labels

For each story:

1. The per-category ensemble scores are divided by a **temperature** (0.7) and
   passed through a sigmoid:
   \[
   p_c = \sigma\left(\\frac{s_c}{T}\\right)
   \]
2. Categories with \( p_c ≥ 0.55 \) are retained as **active labels**, up to a
   maximum of 5 per story.
3. The highest- \( p_c \) among the active labels is the **primary category**.

This yields:

- `primary_category`
- `primary_confidence`
- `all_categories` (multi-label set)
- `num_categories`

#### 5.6 Confidence-Squared Motivation Profiles

To obtain **proportional motivation profiles** (used in the paper’s plots), we
apply **confidence-squared weighting**:

- For each category probability \( p_c \):
  - If \( p_c ≥ 0.55 \): contribution \( = p_c^2 \)
  - Else: contribution \( = 0.3 × p_c^2 \)
- Contributions are normalised to sum to 1.0.

This yields `motivation_profile` (JSON) plus one column per category
(`profile_<CategoryName>`).

---

### 6. Output Format

The output CSV (e.g. `results/motivation_ensemble_v2_results_YYYYMMDD_HHMMSS.csv`)
contains, for each input story:

- Original identifier (`short_name` or your chosen `id_col`)
- `clean_story`, `story_length`, `activity_type` (if present)
- `zero_shot_scores`, `keyword_scores`, `emotion_scores`, `sentiment_scores`
- `all_raw_scores`, `all_probs`
- `primary_category`, `primary_confidence`
- `all_categories` (JSON list), `num_categories`
- `motivation_profile` (JSON dict)
- `profile_<Category>` – one column per motivation category

This is exactly the information used in the Kingston–Bayes study for downstream
validation, plotting, and substantive analysis.

---

### 7. Reproducibility Notes

- The logic in `src/motivation_ensemble_v2.py` is directly derived from the
  **V2 ensemble script** used in the published analysis, but refactored to:
  - Remove hard-coded local paths
  - Expose a clean `run_analysis` function
  - Make the module usable via `python -m`
- The **keyword dictionaries**, **specificity weights**, **model names**, and
  **weighting scheme** exactly match the V2 configuration described in the
  updated NVSQ documentation.

If you need additional components (e.g. the PostgreSQL collection scripts,
validation plots, or donor-level analytics), they can be added here as
additional modules making use of this core ensemble implementation.

