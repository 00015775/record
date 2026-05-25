# UzSL Few-Shot Recognition — Implementation Handoff

Last updated: 2026-05-25.
This document is the single source of truth for picking up the few-shot sign
recognition work. It assumes no prior context — read top-to-bottom and you
should be able to continue improving the model in under 30 minutes.

---

## 1. The 60-second summary

This project is a **few-shot Uzbek Sign Language (UzSL) recognition** system.

- Dataset: **275 isolated signs across 7 topics, 4–5 reps per sign, 1 signer**.
  By design — see "design constraint" below.
- Approach: **ST-GCN encoder + Prototypical Network**. Each sign is stored as the
  mean embedding of its reps. Classification = nearest prototype in cosine space.
- Current honest accuracy (`infer --eval-rep 0`, holding rep-0 out of the DB):
  - **Top-1: 22.2%** (275-way, random would be 0.36%)
  - **Top-5: 56.0%**
  - **Within-topic top-1: 17–69%** depending on topic
- Architecture is correct. Performance is real but underwhelming. **The gap is
  fixable with training-side changes, not data**, see Section 6.

### Why "few-shot" matters
We **cannot** record more reps. The product target is rapid lexicon expansion
(competitive pressure to cover the whole UzSL dictionary), so any approach must
work with 4–5 reps per sign by design. This rules out classical supervised
deep learning at scale — we need few-shot methods, transfer learning, or
template matching. We chose ProtoNet on a learned ST-GCN encoder.

---

## 2. Where everything lives

```
record/
├── app/                  # Recording app (web + terminal UI). Production-ready, not changed often.
├── keypoints/            # Video → keypoints.npz extractor. Stable, don't touch.
├── fewshot/              # ★ This is where you work ★
│   ├── encoder.py        # ST-GCN model (2.05M params, 256-dim output)
│   ├── data.py           # UzSL .npz → (T=64, V=48, 3) joints + Dataset
│   ├── augment.py        # Spatial + temporal augmentations
│   ├── episodes.py       # N-way K-shot sampler (currently RANDOM across all topics)
│   ├── proto.py          # Prototypical loss + cosine/Euclidean similarity
│   ├── pretrained.py     # OpenHands AUTSL weight loader (never had weights to load)
│   ├── train.py          # Episodic training loop with val-rep evaluation
│   ├── prototypes.py     # Build/save/load global prototype DB
│   ├── infer.py          # CLI eval: --npz / --topic / --eval-rep with per-topic breakdown
│   ├── live.py           # Continuous webcam inference (rolling buffer)
│   ├── recognize.py      # Deliberate-capture webcam (SPACE→collect→predict→restart)
│   ├── add_sign.py       # Add a new sign's prototype without retraining
│   ├── config.py         # All hyperparameters, paths, skeleton layout, device auto-detect
│   ├── _mp_utils.py      # Shared MediaPipe→numpy landmark helper
│   ├── proto_db/         # Saved prototype databases (.npz files)
│   ├── pretrained_weights/ # Drop OpenHands ST-GCN weights here if you find them
│   ├── encoder_finetuned.pt        # ★ Current best encoder (val acc 88.4% on 5-way)
│   └── encoder_finetuned_final.pt  # Last-epoch encoder
├── training/             # OLDER pipeline. Superseded by fewshot/. Don't develop here.
├── modeling/             # OLDER pipeline (Conv1D+Transformer). Superseded.
├── inferencing/          # OLDER live inference. Superseded by fewshot/live.py + recognize.py.
├── temp2/                # Reference Kaggle-style code that modeling/ was ported from.
└── Data_Numpy_Arrays_RSL_UzSL/  # Dataset (see Section 3)
```

**You should only modify `fewshot/`.** Everything else is either upstream
(recording / extraction) or legacy. If you see `training/` or `modeling/`
referenced anywhere, treat as historical.

---

## 3. Dataset state (as of handoff)

```
Total .npz files       : 1,107
Unique signs (labels)  : 275
Reps per sign          : 4 (268 classes), 5 (7 classes) — median 4
Topics covered (7)     : Alifbo, Emotsiyalar, Idish-tovoq, Inson Qarindoshlik,
                         Maktab, Sonlar, Tanishish
Signers                : signer01 only
Sequence length        : ~60 frames at 30 fps, normalized to T=64 in pipeline
Resolution             : 1280×720, MediaPipe Holistic complexity=1
```

Each `keypoints.npz` contains: `pose(T,33,4)`, `left_hand(T,21,3)`,
`right_hand(T,21,3)`, `face_full(T,478,3)`, `face_nmm(T,74,3)`, `meta(3,)`.
Few-shot pipeline uses only the **48 joints**: 6 pose (shoulders/elbows/wrists)
+ 21 left hand + 21 right hand. Face is dropped — alphabet/most signs are
hand-dominant.

Labels are stored trilingual (uz/ru/en) in `signs.json` per topic +
`topic_translations.json` at dataset root. Folder paths use the uz name; UI
displays all three.

---

## 4. The model in detail

### 4.1 Skeleton & graph (`fewshot/encoder.py`, `fewshot/config.py`)

48 joints indexed as:
```
0–5   : pose (L_sh=0, R_sh=1, L_el=2, R_el=3, L_wr=4, R_wr=5)
6–26  : left hand (21 MediaPipe Hand landmarks)
27–47 : right hand (21 MediaPipe Hand landmarks)
```

49 edges encoding the kinematic chain + finger topology + pose-wrist→hand-wrist
bridges. Adjacency is built as a 3-partition matrix
(identity / inward-to-root / outward-from-root) following Yan et al. 2018.
Root joint = index 0 (left shoulder).

### 4.2 Architecture (`fewshot/encoder.py`)

```
input  : (B, C=3, T=64, V=48)
data BN over flattened (C·V)
ST-GCN block 1  : 3   → 64    (stride 1 over time)
ST-GCN block 2  : 64  → 64    (stride 1)
ST-GCN block 3  : 64  → 128   (stride 2 → T=32)
ST-GCN block 4  : 128 → 128   (stride 1)
ST-GCN block 5  : 128 → 256   (stride 2 → T=16)
ST-GCN block 6  : 256 → 256   (stride 1)
global average pool over (T, V)
linear(256 → EMBED_DIM=256)
output : (B, 256)
```

Each ST-GCN block = `SpatialGCN` (graph conv with 3 partitions) + `TemporalConv`
(1×kernel-size conv on the time axis) + BN + ReLU + residual.

Params: ~2.05M.

### 4.3 Preprocessing (`fewshot/data.py`)

Per .npz:
1. Extract 48 joints (xyz).
2. Per-frame: translate to shoulder midpoint, divide by shoulder distance.
   Frames where shoulders aren't detected stay raw (rarely a problem).
3. Linear-interpolate to T=64.

For live mode (`preprocess_arrays`), same logic applied to deque-of-frame arrays.

### 4.4 Augmentation (`fewshot/augment.py`)

Applied during training only:
- Gaussian noise (σ=0.01, always)
- XY rotation ±12° (p=0.5)
- Uniform scale [0.9, 1.1] (p=0.5)
- XY translation ±0.05 (p=0.5)
- Horizontal flip with L↔R joint swap (p=0.5)
- Time warp [0.8, 1.2] then resample (p=0.5)

### 4.5 Episodic training (`fewshot/episodes.py`, `fewshot/train.py`)

Currently:
- `N_WAY=5`, `K_SHOT=2`, `Q_QUERY=1` (uses 3 of 3 train reps after val split)
- Classes sampled **uniformly at random** across all 275 — **this is the
  problem** (see Section 6).
- Loss = cross-entropy of cosine-similarity logits between queries and the
  N class prototypes.
- AdamW lr=1e-3, weight decay=1e-4, cosine LR schedule after 200-episode warmup.
- 4000 episodes by default. Took ~29 min on MPS.

### 4.6 Prototype DB (`fewshot/prototypes.py`)

After training:
- For each sign label, encode every available rep, take the mean.
- Save as `.npz` with arrays `labels`, `prototypes (N, 256)`, `counts`, `meta`.
- Default location: `fewshot/proto_db/uzsl_global.npz`.

For honest evaluation, pass `--exclude-rep 0` to skip rep-0 from the DB, then
run `infer --eval-rep 0` to classify rep-0 against the held-out DB.

### 4.7 Inference

| Script | Use case |
|---|---|
| `infer.py --npz <path>` | Classify a single .npz file |
| `infer.py --eval-rep 0` | Honest top-1/top-5 across the whole dataset |
| `infer.py --topic <T>`  | Evaluate one topic's reps |
| `live.py`               | Continuous rolling-buffer webcam |
| `recognize.py`          | Deliberate SPACE-to-record capture (closer to the real UX) |

All scripts accept `--device auto|cpu|mps|cuda`. Auto-detection in
`config.resolve_device()`.

---

## 5. Current performance (honest)

Held-out rep-0 vs prototypes built from reps 1, 2, 3:

```
total: 275
top-1: 22.2%    top-5: 56.0%

per-topic accuracy:
topic                                     reps  top-1  top-5  within-topic top-1
Alifbo                                      29  13.8%  41.4%        17.2%
Emotsiyalar. Tuyg'ular. Holatlar            16  31.2%  50.0%        68.8%
Idish-tovoq. Oziq-ovqat                     17  23.5%  47.1%        58.8%
Inson. Qarindoshlik. Oila                   67  25.4%  55.2%        31.3%
Maktab                                      45  13.3%  57.8%        28.9%
Sonlar                                      72  29.2%  66.7%        30.6%
Tanishish                                   29  13.8%  51.7%        31.0%

most-frequent topic-pair confusions:
  Sonlar    → Sonlar    ×46  (same topic)
  Alifbo    → Alifbo    ×24  (same topic)
  Inson     → Inson     ×23  (same topic)
  Maktab    → Maktab    ×21  (same topic)
  Tanishish → Inson     ×10
```

### Key insight from the per-topic breakdown
The encoder **separates topics well** (almost all confusions are within-topic)
but **fails on within-topic distinctions**. This is exactly what 5-way random
episodic training produces — episodes are usually cross-topic (easy), so the
network never learns to discriminate "Sonlar/21 vs Sonlar/22".

### Live behavior anecdote
In `recognize.py`, predictions feel "stuck on one letter" for the alphabet.
Two compounding causes:
1. Alphabet within-topic acc is only 17.2% — the encoder genuinely doesn't
   separate alphabet letters well.
2. `TEMPERATURE=10` in `proto.similarity` makes softmax extremely peaky, so
   whichever prototype is marginally closest gets ~99% probability regardless
   of true certainty. The UI looks confident even when it's guessing.

---

## 6. Recommended next steps (ranked by ROI)

### Tier 1 — Almost-free changes (do these first, no retraining)

#### 1A. Lower the cosine temperature
File: `fewshot/config.py`, line `TEMPERATURE = 10.0`. Change to `3.0` or `5.0`.
This won't change top-k ordering — only the confidence values shown to users.
Stops the "always 99% on a wrong answer" pathology.

#### 1B. Add an abstain threshold in `recognize.py` / `live.py`
If `top1_p - top2_p < 0.05` → show "—" instead of locking on a label. Tiny
gaps mean the model is essentially guessing. ~5 lines of code.

#### 1C. Make `--topic` the default UX
Within-topic numbers are 31–69% which is dramatically better than 22% global.
The web UI already has a topic selector; mirror that in the live tool so the
user picks a topic before signing. This is a UX choice not a model fix.

### Tier 2 — Training-side changes (the real wins)

#### 2A. Hard-negative episodic sampling — HIGHEST IMPACT
File: `fewshot/episodes.py`. Currently `sample_episode()` picks N labels
uniformly at random. Change to:

```python
# In EpisodeSampler.__init__, group labels by topic
self.labels_by_topic = defaultdict(list)
for lbl in self.labels:
    topic = lbl.split("/")[0]
    self.labels_by_topic[topic].append(lbl)
self.topics_with_enough = [t for t, L in self.labels_by_topic.items()
                            if len(L) >= n_way]

# In sample_episode():
if random() < 0.6 and self.topics_with_enough:
    topic = self.rng.choice(self.topics_with_enough)
    chosen = self.rng.sample(self.labels_by_topic[topic], self.n_way)
else:
    chosen = self.rng.sample(self.labels, self.n_way)
```

Expected lift: **+8 to +12 absolute pts on top-1**.

#### 2B. Increase N_WAY 5 → 20
Just change `config.N_WAY = 20`. Each episode now requires the encoder to
discriminate 20 classes instead of 5 — much closer to deployment difficulty.
Combined with 2A, this is where most of the gain comes from.

Memory note: N=20, K=2, Q=1 → 60 sequences per episode batch.
That's 60 × 3 × 64 × 48 floats = 13 MB; fine for MPS/CUDA, may need to lower
on tiny RAM.

#### 2C. More episodes (4000 → 8000)
Just `--episodes 8000`. Harder objective (2A+2B) needs more iterations.
Time on MPS: ~60 min for 8000 episodes.

#### 2D. L2-normalize the encoder output
File: `fewshot/encoder.py`, in `STGCNEncoder.forward()`:
```python
x = self.head(x)
return F.normalize(x, dim=-1)   # add this line
```
Cleaner ProtoNet formulation. Cosine similarity then degenerates to dot
product. Typical gain +3 to +5 pts. **Note**: existing checkpoints become
inconsistent with this change; you'd need to retrain.

### Tier 3 — Bigger interventions (if Tier 2 isn't enough)

#### 3A. Hand-prominence branch for the alphabet
Right now ST-GCN treats all 48 joints uniformly. Hand details (which dominate
alphabet) get washed out by pose joints (which dominate motion-based signs).
Add a **second branch** in the encoder that operates only on the 42 hand
joints, concatenate its embedding with the body branch.

Sketch:
```python
self.body_branch = ... # current ST-GCN over all 48 joints
self.hand_branch = ... # smaller ST-GCN over 42 hand joints only
# In forward:
body_emb = self.body_branch(x)            # (B, 256)
hand_emb = self.hand_branch(x[:, :, :, 6:])  # (B, 128)
return self.head(torch.cat([body_emb, hand_emb], dim=1))
```

Especially helps alphabet (currently 13.8% top-1). Expected combined gain
across all topics: +5 to +8 pts.

#### 3B. Two-stage classifier (topic → sign)
Stage 1: topic classifier (7 classes, easy, currently the encoder already
nails this).
Stage 2: within-topic sign classifier (max 72 classes, the hard part).
Deploy: predict topic first, then pick prototype from within-topic DB.

This essentially **forces** the 22% number up because each stage is much
smaller. Realistic combined accuracy 55–70% top-1.

Architecture-wise: train one encoder, build per-topic prototype DBs, add a
linear topic head trained as a side-task. Or stay implicit: at inference,
detect which topic the embedding is closest to (mean of per-topic prototypes),
then classify within that topic only.

#### 3C. OpenHands AUTSL pretrained weights
`fewshot/pretrained.py` is wired to load them but no URL has ever been
populated. If you can locate the checkpoint
(https://github.com/AI4Bharat/OpenHands releases), drop it at
`fewshot/pretrained_weights/stgcn_autsl.pth` and rerun training. AUTSL is
Turkish Sign Language — linguistically closest to UzSL. Expected gain:
+5 to +10 pts if the weights map cleanly.

#### 3D. Self-supervised pretraining on the unlabeled video frames
Every recording has 60 frames but we only use 4 reps per sign as supervision.
Train a SimCLR-style contrastive loss on keypoint sequences first (augmented
pairs as positives, random pairs as negatives), then fine-tune episodically.
~1 day of work but the biggest possible ceiling lift if data stays at 4 reps.

### Tier 4 — Architectural/data alternatives (last resort)

- **DTW + k-NN** instead of learned embeddings. Classical, no training needed,
  works well with very few examples. Slower at inference (O(N_classes) per
  prediction) but `tslearn` makes it ~50ms for 275 templates.
- **Add a second signer** even with only a handful of reps. Drastically helps
  generalization. Hard to do if competitive constraints forbid it.

---

## 7. Recommended order of operations

If you have **1 hour**: do Tier 1 (10 min) + run Tier 2A+2B as default config
(50 min retrain). Should land at ~35% top-1, 70% top-5.

If you have **half a day**: above + Tier 2D (L2 norm, retrain again) + Tier 3A
(hand branch). Should land at ~45% top-1, 78% top-5.

If you have **a day or more**: above + Tier 3B (two-stage) or 3D
(self-supervised). 50%+ top-1 is realistic.

---

## 8. Things that didn't work (and why)

- **Older `training/` BiLSTM + handcrafted features**: ~10–15% test on Alifbo
  with k=4 leave-one-out CV. Architecture too generic for skeleton data.
  Superseded by `fewshot/`.

- **Older `modeling/` Conv1D+Transformer (Kaggle-port)**: 13.8% test on Alifbo
  rep-0. Same problem — direct supervised classification with 3 train reps
  per class is hopeless at this data scale.

- **RandomForest baseline showing 100%**: Was trained with `--overfit` flag
  (train=val=test). Real accuracy ~24% on Alifbo. The 100% checkpoints in
  `modeling/checkpoints/baseline_*_rf.pkl` are misleading and should be
  ignored or deleted.

- **OpenHands pretrained init**: Loader works but no checkpoint has been
  obtained. Currently falls back to scratch silently.

---

## 9. Quick commands reference

```bash
# Setup (already done, just for reference)
source .venv/bin/activate

# Train (auto-picks MPS on Mac, CUDA on Linux)
python -m fewshot.train

# Train with overrides
python -m fewshot.train --n-way 20 --episodes 8000 --val-rep 0

# Build prototype DB (use trained encoder)
python -m fewshot.prototypes

# Honest evaluation (hold rep-0 out of DB, classify against it)
python -m fewshot.prototypes --exclude-rep 0 --out fewshot/proto_db/uzsl_no_rep0.npz
python -m fewshot.infer --eval-rep 0 --db fewshot/proto_db/uzsl_no_rep0.npz

# Eval on one topic
python -m fewshot.infer --topic Alifbo

# Live webcam (continuous rolling)
python -m fewshot.live

# Webcam deliberate-capture (SPACE to record)
python -m fewshot.recognize
python -m fewshot.recognize --topic Alifbo

# Add a new sign without retraining (after recording 4-5 reps + extracting keypoints)
python -m fewshot.add_sign --topic Tanishish --sign Salom
```

---

## 10. Decisions log (for context)

| Date | Decision | Reasoning |
|---|---|---|
| 2026-05-23 | Build `fewshot/` (ST-GCN + ProtoNet) | Best-known approach for "4-5 reps per sign, growing lexicon" |
| 2026-05-23 | 48 joints (no face) | Hand-dominant signs; face stream noisy at this scale |
| 2026-05-23 | Shoulder-centered normalization | Standard for skeleton recognition; works for any signer position |
| 2026-05-23 | K_SHOT=2 (not 4) | Each class has only 3 train reps after val split |
| 2026-05-24 | MPS auto-device | 5.8× forward speedup; 1.4× end-to-end (CPU dataloader bottleneck) |
| 2026-05-24 | Cosine sim with T=10 | Standard ProtoNet recipe; T=10 turned out too peaky (see Tier 1A) |
| 2026-05-24 | Trained 4000 episodes | First baseline; reached val acc 88.4% on 5-way, 22% on 275-way |

---

## 11. Open questions for the next session

1. Does `recognize.py` actually fix the live UX or is the "stuck on one
   letter" issue more fundamental? Need to retest after Tier 1 fixes.
2. Are we sure `MediaPipe Holistic` is producing the same keypoint
   distribution at recording-time vs live-time? Quick sanity check: record a
   sign live, save the keypoints, compare with an actual recorded .npz of the
   same sign. Distance should be small.
3. Should we add face landmarks (74 NMM) back for non-alphabet signs that
   include facial expressions ("Yes", "No", emotional words)? Topic-aware
   feature selection might help.
4. Is there a path to multi-signer data without burning competitive runway?
   Even 1 additional signer with 2 reps per sign across half the vocabulary
   would help massively.

---

## 12. Contact / context

The current owner is iterating in `fewshot/` only. The legacy directories
(`training/`, `modeling/`, `inferencing/`, `infer/`, `temp/`, `temp2/`) should
be considered frozen reference material. Do not develop there.

For dataset growth, see `app/mod06_webapp.py` (web UI) — recording UX is
already production-quality and shouldn't need changes.

Good luck.
