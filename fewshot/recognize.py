"""Deliberate-capture sign recognition on the webcam.

Workflow:
  IDLE       → press SPACE to start
  RECORDING  → frames stream into the buffer while at least one hand is visible
               when buffer reaches T_FIXED frames → auto-predict
  DONE       → top-5 prediction shown
               press R or SPACE for a new attempt, C to clear, Q to quit

Run:
    python -m fewshot.recognize
    python -m fewshot.recognize --buffer 48 --threshold 0.3
    python -m fewshot.recognize --camera 1 --topic Alifbo   # restrict DB to a topic
"""2
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from fewshot import config
from fewshot._mp_utils import landmark_array
from fewshot.data import preprocess_arrays, to_tensor
from fewshot.encoder import build_encoder
from fewshot.prototypes import load_database
from fewshot.proto import similarity


# ── Colors ────────────────────────────────────────────────────────────────────
WHITE   = (255, 255, 255)
GREY    = (170, 170, 170)
DARK    = (40, 40, 40)
RED     = (60, 80, 230)
GREEN   = (90, 220, 130)
PURPLE  = (210, 150, 90)        # BGR
YELLOW  = (80, 220, 240)
BLUE    = (220, 130, 0)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_panel(frame, x, y, w, h, alpha=0.55, color=DARK):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_top_bar(frame, state: str, frames: int, max_frames: int,
                 has_lh: bool, has_rh: bool):
    h, w = frame.shape[:2]
    _draw_panel(frame, 0, 0, w, 56, alpha=0.65)

    # State chip on the left
    state_text = {"idle": "IDLE", "recording": "RECORDING",
                  "done": "DONE", "no_hands": "WAITING FOR HAND"}.get(state, state.upper())
    state_color = {"idle": GREY, "recording": RED, "done": GREEN,
                   "no_hands": YELLOW}.get(state, WHITE)
    cv2.rectangle(frame, (12, 14), (16, 42), state_color, -1)
    cv2.putText(frame, state_text, (28, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_color, 2)

    # Frame counter (centered)
    counter = f"Frames: {frames:>3d} / {max_frames}"
    (tw, _), _ = cv2.getTextSize(counter, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, counter, (w // 2 - tw // 2, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

    # Hand indicators on the right
    def _chip(text, on, x):
        color = GREEN if on else GREY
        cv2.putText(frame, text, (x, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    _chip("L", has_lh, w - 70)
    _chip("R", has_rh, w - 38)


def draw_controls(frame):
    h, w = frame.shape[:2]
    _draw_panel(frame, 0, h - 36, w, 36, alpha=0.65)
    txt = "SPACE start | C clear | R restart | Q quit"
    cv2.putText(frame, txt, (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)


def draw_progress_bar(frame, frames: int, max_frames: int):
    h, w = frame.shape[:2]
    y = h - 42
    bar_h = 4
    fill = int(w * min(1.0, frames / max(1, max_frames)))
    cv2.rectangle(frame, (0, y), (w, y + bar_h), DARK, -1)
    cv2.rectangle(frame, (0, y), (fill, y + bar_h), PURPLE, -1)


def draw_idle_hint(frame):
    h, w = frame.shape[:2]
    text = "Press  SPACE  to record"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cx, cy = w // 2 - tw // 2, h // 2 - 6
    _draw_panel(frame, cx - 24, cy - th - 18, tw + 48, th + 38, alpha=0.55)
    cv2.putText(frame, text, (cx, cy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2)


def draw_prediction_card(frame, top_k: list[tuple[str, float]], meta: dict):
    """Render the top-5 prediction card with trilingual labels for top-1."""
    h, w = frame.shape[:2]
    card_w, card_h = 380, 230
    x, y = 24, 80
    _draw_panel(frame, x, y, card_w, card_h, alpha=0.78)
    cv2.rectangle(frame, (x, y), (x + card_w, y + card_h), GREEN, 2)

    # Top-1
    top1_lbl, top1_p = top_k[0]
    m = meta.get(top1_lbl, {})
    uz = m.get("sign_uz", top1_lbl.split("/")[-1])
    ru = m.get("sign_ru", "")
    en = m.get("sign_en", "")
    cv2.putText(frame, f"★ {uz}", (x + 16, y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 2)
    cv2.putText(frame, f"{ru}  |  {en}", (x + 16, y + 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1)
    cv2.putText(frame, f"{top1_p*100:.1f}%", (x + 16, y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.line(frame, (x + 12, y + 112), (x + card_w - 12, y + 112), DARK, 1)

    # Top-2..5
    for i, (lbl, p) in enumerate(top_k[1:5], start=2):
        m2 = meta.get(lbl, {})
        uz2 = m2.get("sign_uz", lbl.split("/")[-1])
        row_y = y + 112 + (i - 1) * 24
        cv2.putText(frame, f"{i}. {uz2[:24]}", (x + 16, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        cv2.putText(frame, f"{p*100:5.1f}%", (x + card_w - 80, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1)


def draw_landmarks(frame, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0,0,180), thickness=2))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0,180,0), thickness=2))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                  mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(200,200,200), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(140,140,140), thickness=1))


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",       type=Path, default=config.DEFAULT_PROTO)
    ap.add_argument("--encoder",  type=Path, default=config.ENCODER_CKPT)
    ap.add_argument("--device",   default="auto")
    ap.add_argument("--camera",   type=int, default=0)
    ap.add_argument("--width",    type=int, default=1280)
    ap.add_argument("--height",   type=int, default=720)
    ap.add_argument("--buffer",   type=int, default=config.T_FIXED,
                    help=f"Frames to collect before predicting (default: {config.T_FIXED})")
    ap.add_argument("--top",      type=int, default=5)
    ap.add_argument("--topic",    default=None,
                    help="Restrict predictions to one topic (e.g. 'Alifbo')")
    args = ap.parse_args()

    device = config.resolve_device(args.device)
    if not args.db.exists():
        print(f"prototype DB not found: {args.db}\n"
              f"run `python -m fewshot.prototypes` first")
        return

    db = load_database(args.db)
    all_labels = db["labels"]
    all_protos = db["prototypes"]

    # Optional topic restriction → keep only labels whose key starts with topic+"/"
    if args.topic:
        keep = [i for i, lbl in enumerate(all_labels) if lbl.startswith(args.topic + "/")]
        if not keep:
            print(f"no prototypes in topic '{args.topic}' inside {args.db.name}")
            return
        labels = [all_labels[i] for i in keep]
        prototypes_np = all_protos[keep]
        print(f"restricted to topic '{args.topic}' ({len(labels)} signs)")
    else:
        labels = list(all_labels)
        prototypes_np = all_protos
    prototypes = torch.from_numpy(prototypes_np).to(device)

    encoder = build_encoder().to(device).eval()
    if args.encoder.exists():
        state = torch.load(args.encoder, map_location=device, weights_only=False)
        sd = state.get("encoder_state", state)
        encoder.load_state_dict(sd, strict=False)
        print(f"encoder ← {args.encoder.name}")
    else:
        print(f"WARNING: encoder not found at {args.encoder} (using random init)")
    print(f"device: {device}   prototypes: {len(labels)}")

    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
    )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("cannot open camera"); return

    pose_buf = deque(maxlen=args.buffer)
    lh_buf   = deque(maxlen=args.buffer)
    rh_buf   = deque(maxlen=args.buffer)
    state = "idle"
    top_k: list[tuple[str, float]] = []

    print("\nSPACE start | C clear | R restart | Q quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)

        pose = landmark_array(results.pose_landmarks, 33, with_vis=True)
        lh   = landmark_array(results.left_hand_landmarks, 21)
        rh   = landmark_array(results.right_hand_landmarks, 21)
        has_lh = results.left_hand_landmarks is not None
        has_rh = results.right_hand_landmarks is not None
        has_any = has_lh or has_rh

        # ── State transitions ─────────────────────────────────────────────────
        if state == "recording":
            display_state = "recording" if has_any else "no_hands"
            if has_any:
                pose_buf.append(pose); lh_buf.append(lh); rh_buf.append(rh)
            if len(pose_buf) >= args.buffer:
                # Auto-predict
                pose_arr = np.stack(pose_buf)
                lh_arr   = np.stack(lh_buf)
                rh_arr   = np.stack(rh_buf)
                joints = preprocess_arrays(pose_arr, lh_arr, rh_arr)
                x = to_tensor(joints).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = encoder(x)
                    logits = similarity(emb, prototypes)
                    probs = torch.softmax(logits, dim=-1)[0].cpu()
                top_n = min(args.top, probs.numel())
                top = probs.topk(top_n)
                top_k = [(labels[i], float(p))
                         for p, i in zip(top.values.tolist(), top.indices.tolist())]
                state = "done"
                print(f"  → {top_k[0][0]}  ({top_k[0][1]*100:.1f}%)")
        else:
            display_state = state

        # ── Draw ──────────────────────────────────────────────────────────────
        draw_landmarks(frame, results)
        draw_top_bar(frame, display_state, len(pose_buf), args.buffer, has_lh, has_rh)
        draw_progress_bar(frame, len(pose_buf), args.buffer)
        draw_controls(frame)
        if state == "idle":
            draw_idle_hint(frame)
        if state == "done" and top_k:
            draw_prediction_card(frame, top_k, db["meta"])

        cv2.imshow("fewshot UzSL — capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            # SPACE starts a new recording from any state
            pose_buf.clear(); lh_buf.clear(); rh_buf.clear()
            top_k = []
            state = "recording"
        elif key == ord('c'):
            pose_buf.clear(); lh_buf.clear(); rh_buf.clear()
            top_k = []
            state = "idle"
        elif key == ord('r'):
            pose_buf.clear(); lh_buf.clear(); rh_buf.clear()
            top_k = []
            state = "idle"

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()


if __name__ == "__main__":
    main()
