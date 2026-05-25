"""Live webcam inference: MediaPipe Holistic → ST-GCN encoder → prototype DB.

Run:
    python -m fewshot.live
    python -m fewshot.live --camera 1 --top 3 --hands-required
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from fewshot import config
from fewshot._mp_utils import landmark_array as _landmark_array
from fewshot.data import preprocess_arrays, to_tensor
from fewshot.encoder import build_encoder
from fewshot.prototypes import load_database
from fewshot.proto import similarity


def _load_encoder(ckpt_path: Path, device: torch.device):
    enc = build_encoder().to(device).eval()
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = state.get("encoder_state", state)
        enc.load_state_dict(sd, strict=False)
    return enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",       type=Path, default=config.DEFAULT_PROTO)
    ap.add_argument("--encoder",  type=Path, default=config.ENCODER_CKPT)
    ap.add_argument("--device",   default="auto")
    ap.add_argument("--camera",   type=int, default=0)
    ap.add_argument("--width",    type=int, default=1280)
    ap.add_argument("--height",   type=int, default=720)
    ap.add_argument("--buffer",   type=int, default=config.T_FIXED,
                    help=f"Frames to buffer (default: {config.T_FIXED})")
    ap.add_argument("--predict-every", type=float, default=0.4,
                    help="Seconds between predictions (default: 0.4)")
    ap.add_argument("--top",      type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.4,
                    help="Min top-1 prob to show prediction (default: 0.4)")
    ap.add_argument("--smooth",    type=int, default=3,
                    help="Majority-vote over last N predictions (default: 3)")
    ap.add_argument("--hands-required", action="store_true",
                    help="Pause inference when no hands detected")
    args = ap.parse_args()

    device = config.resolve_device(args.device)
    if not args.db.exists():
        print(f"prototype DB not found: {args.db}\n"
              f"run `python -m fewshot.prototypes` first")
        return

    db = load_database(args.db)
    labels = db["labels"]
    prototypes = torch.from_numpy(db["prototypes"]).to(device)
    encoder = _load_encoder(args.encoder, device)
    print(f"loaded {len(labels)} prototypes from {args.db.name}")
    print(f"device: {device}")

    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("cannot open camera"); return

    pose_buf  = deque(maxlen=args.buffer)
    lh_buf    = deque(maxlen=args.buffer)
    rh_buf    = deque(maxlen=args.buffer)
    pred_hist = deque(maxlen=args.smooth)

    last_pred_time = 0.0
    current_pred: list[tuple[str, float]] = []   # top-k list
    hand_missing_count = 0

    print("\npress  q quit  |  c clear buffer  |  space force-predict\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)

        # Build per-frame landmark arrays
        pose = _landmark_array(results.pose_landmarks, 33, with_vis=True)
        lh   = _landmark_array(results.left_hand_landmarks, 21)
        rh   = _landmark_array(results.right_hand_landmarks, 21)

        has_lh = results.left_hand_landmarks is not None
        has_rh = results.right_hand_landmarks is not None
        has_any_hand = has_lh or has_rh

        # Hand-presence gating
        if args.hands_required:
            if has_any_hand:
                hand_missing_count = 0
                pose_buf.append(pose); lh_buf.append(lh); rh_buf.append(rh)
            else:
                hand_missing_count += 1
                if hand_missing_count > 5:
                    pose_buf.clear(); lh_buf.clear(); rh_buf.clear()
                    current_pred = []
                    pred_hist.clear()
        else:
            pose_buf.append(pose); lh_buf.append(lh); rh_buf.append(rh)

        # Periodic prediction
        now = time.time()
        if len(pose_buf) == args.buffer and now - last_pred_time >= args.predict_every:
            pose_arr = np.stack(pose_buf)
            lh_arr   = np.stack(lh_buf)
            rh_arr   = np.stack(rh_buf)
            joints = preprocess_arrays(pose_arr, lh_arr, rh_arr)
            x = to_tensor(joints).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = encoder(x)
                logits = similarity(emb, prototypes)
                probs = torch.softmax(logits, dim=-1)[0].cpu()
            top_k = min(args.top, probs.numel())
            top = probs.topk(top_k)
            current_pred = [(labels[i], float(p))
                            for p, i in zip(top.values.tolist(), top.indices.tolist())]
            pred_hist.append(current_pred[0][0])
            last_pred_time = now

        # Draw landmarks
        rgb.flags.writeable = True
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
                                      mp_drawing.DrawingSpec(color=(220,220,220), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(150,150,150), thickness=1))

        # HUD
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 110), (0, 0, 0), -1)
        if current_pred:
            top1_label, top1_p = current_pred[0]
            meta = db["meta"].get(top1_label, {})
            disp_uz = meta.get("sign_uz", top1_label.split("/")[-1])
            disp_ru = meta.get("sign_ru", "")
            disp_en = meta.get("sign_en", "")

            # Smoothed top-1 (majority vote)
            smoothed = max(set(pred_hist), key=lambda x: pred_hist.count(x)) if pred_hist else top1_label
            if smoothed != top1_label:
                # show smoothed display
                meta2 = db["meta"].get(smoothed, {})
                disp_uz = meta2.get("sign_uz", smoothed.split("/")[-1])
                disp_ru = meta2.get("sign_ru", "")
                disp_en = meta2.get("sign_en", "")

            if top1_p >= args.threshold:
                cv2.putText(frame, f"{disp_uz}", (12, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(frame, f"{disp_ru} | {disp_en}", (12, 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
                cv2.putText(frame, f"{top1_p*100:.0f}%   top: " +
                            ", ".join(f"{l.split('/')[-1]}={p*100:.0f}%" for l,p in current_pred[1:]),
                            (12, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
            else:
                cv2.putText(frame, "—",  (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2)
                cv2.putText(frame, f"(top1 {top1_p*100:.0f}% < {args.threshold*100:.0f}%)",
                            (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140,140,140), 1)
        else:
            cv2.putText(frame, "filling buffer …", (12, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        buf_w = int(w * (len(pose_buf) / args.buffer))
        cv2.rectangle(frame, (0, h-6), (buf_w, h), (124, 106, 247), -1)

        cv2.imshow("fewshot UzSL", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            pose_buf.clear(); lh_buf.clear(); rh_buf.clear()
            pred_hist.clear(); current_pred = []
        if key == ord(' ') and len(pose_buf) == args.buffer:
            last_pred_time = 0  # force re-predict next frame

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()


if __name__ == "__main__":
    main()
