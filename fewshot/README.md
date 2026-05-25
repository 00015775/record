# fewshot — UzSL Isolated-Sign Recognition (Prototypical Network + ST-GCN)

A few-shot sign recognition system designed for **4–5 reps per sign** across a
growing vocabulary. Built on two ideas:

1. **ST-GCN encoder** — graph conv over the kinematic skeleton (48 joints:
   6 pose + 2×21 hands). Learns "what makes signs similar/different" once.
2. **Prototypical Network** — each sign is stored as the mean embedding of its
   recorded reps. Classification = find the closest prototype.

The encoder is fine-tuned **episodically**: each training step picks N random
signs and learns to discriminate them from 4 examples each. This is the exact
operating mode of deployment.

## Why this approach for our data

| Property | Why it fits |
|---|---|
| Only 4–5 reps per sign | ProtoNet's native operating point |
| Vocabulary grows over time | Add new sign → encode 4 reps → store mean. **No retraining.** |
| Single signer at first | Episodic loss doesn't care, encoder still generalises |
| Real-time inference target | Encoder runs once per buffer; comparison to 2000 prototypes ≈ 2 ms |

## Quick start

```bash
# 1. Episodic fine-tune the encoder on existing UzSL data
python -m fewshot.train --device cpu

# 2. Build the global prototype DB
python -m fewshot.prototypes --device cpu

# 3. Evaluate on held-out reps
python -m fewshot.infer --eval-rep 0

# 4. Live webcam
python -m fewshot.live --hands-required

# 5. After recording a new sign, add it (no retraining!)
python -m fewshot.add_sign --topic Tanishish --sign Salom
```

## Files

| File | Purpose |
|---|---|
| `config.py`     | All hyperparameters in one place |
| `encoder.py`    | ST-GCN encoder (skeleton graph + temporal conv) |
| `data.py`       | UzSL → 48-joint (T, V, 3) adapter + Dataset |
| `augment.py`    | Spatial + temporal augmentations |
| `episodes.py`   | N-way K-shot episode sampler |
| `proto.py`      | Prototype loss + cosine/Euclidean similarity |
| `pretrained.py` | Best-effort loader for OpenHands AUTSL weights (falls back to scratch) |
| `train.py`      | Episodic training loop |
| `prototypes.py` | Build / save / load prototype DB |
| `infer.py`      | CLI for `.npz` or topic-wide evaluation |
| `live.py`       | Webcam inference with smoothing + confidence gate |
| `add_sign.py`   | Extend DB with a new sign without retraining |

## Architecture details

### Skeleton (48 joints, see `config.py`)

```
0–5   : pose arms      (L_sh, R_sh, L_el, R_el, L_wr, R_wr)
6–26  : left hand      (21 MediaPipe Hand landmarks)
27–47 : right hand     (21 MediaPipe Hand landmarks)
```

Edges encode the kinematic chain (shoulders→elbows→wrists→hand-wrists, plus
finger topology). The 3-partition adjacency (identity / inward / outward) is
the canonical Yan-et-al. setup, so OpenHands' AUTSL pretrained weights can
load with shape-matched keys.

### Normalisation

Per-frame translate to **shoulder midpoint** and scale by **shoulder distance**.
Frames where shoulders aren't detected are left as zero — masked downstream by
the encoder's data-BN.

### Augmentation (training only)

XY rotation ±12°, scale ±10%, translation ±0.05, horizontal flip (which swaps
left/right joints across the body axis), time warp 0.8–1.2×, Gaussian noise.

### Episodic loss

```
For each step:
  N classes × (K support + Q query) = N(K+Q) sequences encoded
  prototypes = mean of K support embeddings per class
  logits[q] = temperature * cos_sim(query_q, prototypes)
  loss      = cross_entropy(logits, true_class)
```

### Pretrained init

`pretrained.py` looks under `pretrained_weights/` for any of:
`stgcn_autsl.pth`, `stgcn.pth`, `encoder.pth`.  If you have a OpenHands
checkpoint, drop it there and it will be loaded by shape-matched key rename.
Otherwise training proceeds from scratch with no error.

## Adding a new sign in production

1. Record 4–5 reps via the web UI (`python app/mod06_webapp.py`)
2. Extract keypoints (`python keypoints/extract_keypoints.py --topic X --signer Y`)
3. Add to DB (`python -m fewshot.add_sign --topic X --sign NewSign --signer Y`)
4. Restart `python -m fewshot.live` — the new sign is now recognised.

No retraining required.
