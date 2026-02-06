
# VOICE DETECTOR API

# 1️⃣ Evolution of the System

## 🔹 Phase 1 – LGBM LoLo Baseline Model

The first version of the system used a **LightGBM (LGBM) classifier** trained on low-level handcrafted acoustic features.

### 📊 Performance

* Accuracy: **~85%**
* Weakness: Failed on **high-quality studio recordings**
* Training bias: Mostly low-quality, noisy recordings
* Could not generalize to:

  * Clean AI
  * Studio human voices
  * Augmented / compressed audio

This version served as a strong baseline but exposed the need for:

* Better representation learning
* Robustness to degradation
* AI artifact detection

---

# 2️⃣ Phase 2 – Hand-Crafted Human / AI Feature Enhancements

To fix baseline weaknesses, we introduced signal-level feature engineering.

### 🧠 Human-Degraded Detection

We defined degraded audio using:

#### Signal-to-Noise Ratio (SNR)

[
SNR_{dB} = 20 \log_{10} \left( \frac{RMS}{NoiseFloor} \right)
]

Where:

* ( RMS = \sqrt{\frac{1}{N} \sum x^2} )
* NoiseFloor = 5th percentile amplitude

Degraded if:

* SNR < 18 dB

---

#### Spectral Flatness

[
Flatness = \frac{\exp(\text{mean}(\log(P)))}{\text{mean}(P)}
]

Where:

* (P) = power spectrum

Higher flatness → more noise-like
Lower flatness → tonal/studio

---

#### Clipping Ratio

[
Clipped = \frac{#(|x| > 0.99)}{N}
]

Used to detect distortion.

---

### 🤖 AI Artifact Detection

We engineered AI-specific indicators:

* **Repetition Score**

  * MFCC cosine similarity across chunks
  * High similarity → looped patterns

* **Pitch Variance**

  * Estimated F0 via autocorrelation
  * Low variance → synthetic

* **Vocoder Artifact Score**
  [
  Score = Flatness + 0.02(1 - HF_{ratio})
  ]

* **Dynamics Ratio**
  [
  dyn = \frac{\sigma(RMS)}{\mu(RMS)}
  ]

Low dynamics → monotone AI

---

This improved robustness, but classical ML still lacked deep representation power.

---

# 3️⃣ Phase 3 – Stage 1 Deep Model (Wav2Vec2)

We upgraded to a **representation learning model**:

### Architecture

* `facebook/wav2vec2-base`
* Mean pooled embeddings
* MLP head:

  * 768 → 256 → 1

### Stage 1 Role

Binary classifier:

* Outputs probability of AI

### Stage 1 Constraints & Thresholds

```python
S1_HUMAN_RECHECK_THRESHOLD = 0.40
S1_AI_CHECK_THRESHOLD = 0.75
S1_VERY_CONFIDENT = 0.90
S1_CONFIDENT = 0.82
```

### Decision Logic

| Range     | Meaning                     |
| --------- | --------------------------- |
| < 0.40    | HUMAN (trusted immediately) |
| 0.40–0.75 | Ambiguous                   |
| > 0.75    | AI candidate                |
| > 0.90    | Very confident AI           |

Stage 1 became strong — but occasionally overconfident.

So we built Stage 2.

---

# 4️⃣ Stage 2 – AASIST Advanced Verifier

## What is AASIST?

AASIST is a **graph attention-based backend** that analyzes:

* Frame-level temporal patterns
* Spectral artifacts
* Subtle vocoder cues
* Cross-frame dependencies

Architecture:

```
Wav2Vec2 Encoder → AASIST Backend → Classifier
```

---

# 🧠 Core Innovation

## Confidence-Weighted Adaptive Verification

Instead of hard thresholds:

> Different confidence levels require different strategies.

---

# 🎯 Decision Flow

---

## Case 1: Stage 1 says HUMAN (< 0.40)

✔ Trust immediately
✔ No AASIST verification
✔ Minimizes false AI labeling

---

## Case 2: Stage 1 says AI (> 0.75)

Tiered adaptive logic:

---

###  TIER 1: Very Confident AI (> 0.90)

| AASIST    | Decision                 |
| --------- | ------------------------ |
| ≥ 0.40    | AI (70% S1 + 30% AASIST) |
| 0.20–0.40 | AI (trust S1)            |
| < 0.20    | HUMAN                    |

---

###  TIER 2: Confident AI (0.82–0.90)

| AASIST    | Decision                |
| --------- | ----------------------- |
| ≥ 0.45    | AI (65/35 weighted)     |
| 0.25–0.45 | Weighted check          |
| < 0.25    | Feature-based tie break |

---

###  TIER 3: Moderate AI (0.75–0.82)

| AASIST    | Decision     |
| --------- | ------------ |
| ≥ 0.50    | AI (50/50)   |
| 0.30–0.50 | INCONCLUSIVE |
| < 0.30    | HUMAN        |

---

## Case 3: Stage 1 Ambiguous (0.40–0.75)

Trust AASIST more:

[
FinalScore = 0.4 \cdot S1 + 0.6 \cdot AASIST
]

| AASIST    | Decision          |
| --------- | ----------------- |
| ≥ 0.55    | AI                |
| < 0.35    | HUMAN             |
| 0.35–0.55 | Weighted decision |

---

# 📊 Why This Is Powerful

### ✅ Reduces False Positives

Human voices rarely cross S1 < 0.40.

### ✅ Reduces False Negatives

Very confident S1 AI (>0.90) no longer blocked by mild AASIST uncertainty.

### ✅ Handles Studio Audio

Hand-crafted features compensate for over-clean recordings.

### ✅ Handles Augmented AI

Vocoder + repetition detection catches artifacts.

---

# 📈 Example

Example:

```
S1 = 0.85
AASIST = 0.38
```

Weighted:

[
0.6(0.85) + 0.4(0.38) = 0.662
]

→ AI

Old system → INCONCLUSIVE
New system → Correct AI

---

# 🏗 Architecture Overview

```
Audio Input
    ↓
Chunking (4s, 50% overlap)
    ↓
Stage 1 (Wav2Vec2 + MLP)
    ↓
Hand-crafted feature extraction
    ↓
AASIST (conditional verification)
    ↓
Confidence-weighted decision fusion
    ↓
Final Label + Explanation
```

---

# 🔍 Explanation Engine

The system produces structured layman explanations based on:

* Breathing detection
* Pitch variance
* Dynamics ratio
* Repetition score
* Vocoder artifacts
* Clipping
* Reverb
* Micro-noises

This makes the system explainable — critical for hackathon judging.

---

# 🚀 Final System Characteristics

| Feature           | Supported    |
| ----------------- | ------------ |
| Studio Human      | ✅            |
| Low Quality Human | ✅            |
| Augmented AI      | ✅            |
| Clean AI          | ✅            |
| Edge Cases        | INCONCLUSIVE |
| Confidence Output | Yes          |
| Explainability    | Yes          |

---

It is a **multi-stage adaptive AI verification system** built to handle real-world edge cases.

