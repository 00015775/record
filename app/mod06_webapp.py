"""
Web UI for UzSL data collection.

Run:  python app/mod06_webapp.py
Browser opens automatically at http://127.0.0.1:5000

The camera feed is streamed as MJPEG directly into the browser — no CV2 window.
Recording state machine runs in a background thread.
"""
import json
import shutil
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

import re

import cv2
from flask import Flask, Response, jsonify, redirect, render_template, request, send_file, session, url_for

from mod01_config import FPS, FRAME_HEIGHT, FRAME_WIDTH
from translations import TRANSLATIONS
from mod02_storage import (
    add_sign, add_topic, count_all_repetitions, count_repetitions,
    list_signers, load_sign_list, load_topic_list, load_topic_translations,
    path_videos, recorded_signs, sign_uz,
)

app = Flask(__name__, template_folder='../templates')
app.secret_key = "uzsl-recorder-lang-key"


# ── Frame buffer (camera thread → stream endpoint) ────────────────────────────

class FrameBuffer:
    def __init__(self):
        self._cond      = threading.Condition(threading.Lock())
        self._frame     = None
        self._seq       = 0
        self._last_time = 0.0

    def put(self, frame):
        with self._cond:
            self._frame     = frame.copy()
            self._seq      += 1
            self._last_time = time.time()
            self._cond.notify_all()

    def get(self):
        with self._cond:
            return None if self._frame is None else self._frame.copy()

    def get_next(self, last_seq: int):
        """Block until a frame newer than last_seq is available (max 1 s wait).
        Returns (frame_copy, new_seq)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq, timeout=1.0)
            if self._frame is None:
                return None, last_seq
            return self._frame.copy(), self._seq

    def is_alive(self) -> bool:
        with self._cond:
            return (time.time() - self._last_time) < 2.0


frame_buf = FrameBuffer()


# ── Camera helpers ────────────────────────────────────────────────────────────

def _macos_camera_names() -> list[str]:
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        return [c.get("_name", "") for c in json.loads(out).get("SPCameraDataType", [])]
    except Exception:
        return []


def scan_cameras() -> list[dict]:
    """List cameras using system_profiler (no cv2 probing, so the active camera is never disturbed)."""
    names = _macos_camera_names()
    if names:
        return [{"index": i, "name": n or f"Camera {i}"} for i, n in enumerate(names)]
    # Fallback: probe with cv2 only when system_profiler returns nothing
    cams = []
    for idx in range(8):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.release()
            cams.append({"index": idx, "name": f"Camera {idx}"})
        elif cams:
            break
    return cams or [{"index": 0, "name": "Camera 0"}]


# ── Shared state ──────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self._lock = threading.Lock()
        # navigation
        self.signer_id: str | None = None
        self.topic:     str | None = None
        self.sign:      str | None = None
        self.rep_idx:   int        = 0
        # recording lifecycle
        self.cmd:            str | None = None   # 'record'
        self.recording:      bool       = False
        self.countdown_secs: int        = 3
        self.countdown_end:  float      = 0.0
        self.stop_requested: bool       = False
        self.frame_count:    int        = 0
        self.awaiting_decision: bool    = False
        self.decision:       str | None = None   # 'save' | 'discard'
        self.last_result:    str | None = None   # 'success'|'discarded'|'error'
        self.error:          str | None = None
        # camera
        self.camera_idx:     int        = 0
        self.pending_camera: int | None = None
        self.cameras:        list[dict] = []
        # control
        self.running: bool = True

    def set(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def status_dict(self) -> dict:
        with self._lock:
            now = time.time()
            if self.recording:
                ph = "countdown" if now < self.countdown_end else "recording"
            elif self.awaiting_decision:
                ph = "awaiting_decision"
            elif self.last_result:
                ph = "done"
            else:
                ph = "idle"
            return {
                "phase":       ph,
                "last_result": self.last_result,
                "rep_idx":     self.rep_idx,
                "frame_count": self.frame_count,
                "error":       self.error,
            }


state = AppState()


# ── Decision helper ───────────────────────────────────────────────────────────

def _apply_decision(rep_dir: Path, save: bool) -> None:
    if save:
        state.set(rep_idx=state.rep_idx + 1, last_result="success",
                  awaiting_decision=False, decision=None)
    else:
        shutil.rmtree(rep_dir, ignore_errors=True)
        state.set(last_result="discarded", awaiting_decision=False, decision=None)


# ── Context processor ─────────────────────────────────────────────────────────

@app.context_processor
def inject_globals():
    lang = session.get("lang", "en")
    return {
        "cameras": state.cameras,
        "current_camera": state.camera_idx,
        "tr": TRANSLATIONS[lang],
        "lang": lang,
    }

@app.route("/set_lang")
def set_lang():
    lang = request.args.get("lang", "en")
    if lang in TRANSLATIONS:
        session["lang"] = lang
    return redirect(request.referrer or url_for("signer_page"))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("signer_page"))

# — Signer ——————————————————————————————————————————————————————————————————

@app.route("/signer")
def signer_page():
    return render_template("signer.html", signers=list_signers())

@app.route("/signer/set", methods=["POST"])
def signer_set():
    sid = request.form.get("signer_id", "").strip()
    if sid.startswith("signer") and sid[6:].isdigit():
        state.set(signer_id=sid, topic=None, sign=None, rep_idx=0)
        return redirect(url_for("topic_page"))
    return render_template("signer.html", signers=list_signers(),
                           error="signer_format_error")

# — Topic ———————————————————————————————————————————————————————————————————

@app.route("/topic")
def topic_page():
    if not state.signer_id:
        return redirect(url_for("signer_page"))
    topics = load_topic_list()
    translations = load_topic_translations()
    topic_data = [
        {"uz": t,
         "ru": translations.get(t, {}).get("ru", t),
         "en": translations.get(t, {}).get("en", t),
         "total": len(load_sign_list(t)),
         "recorded": len(recorded_signs(t, state.signer_id))}
        for t in topics
    ]
    return render_template("topic.html", signer_id=state.signer_id, topic_data=topic_data)

@app.route("/topic/set", methods=["POST"])
def topic_set():
    topic = request.form.get("topic", "").strip()
    if topic:
        state.set(topic=topic, sign=None, rep_idx=0)
    return redirect(url_for("sign_page"))

@app.route("/topic/add", methods=["POST"])
def topic_add():
    new = request.form.get("new_topic", "").strip()
    if new:
        add_topic(new)
    return redirect(url_for("topic_page"))

# — Sign ————————————————————————————————————————————————————————————————————

@app.route("/sign")
def sign_page():
    if not state.signer_id:
        return redirect(url_for("signer_page"))
    if not state.topic:
        return redirect(url_for("topic_page"))
    signs = load_sign_list(state.topic)
    rep_counts = count_all_repetitions(state.topic, state.signer_id)
    sign_data = [
        {"uz": s["uz"], "ru": s["ru"], "en": s["en"],
         "reps": rep_counts.get(s["uz"], 0), "recorded": s["uz"] in rep_counts}
        for s in signs
    ]
    translations = load_topic_translations()
    topic_full = {
        "uz": state.topic,
        "ru": translations.get(state.topic, {}).get("ru", state.topic),
        "en": translations.get(state.topic, {}).get("en", state.topic),
    }
    return render_template("sign.html",
                           signer_id=state.signer_id,
                           topic=state.topic,
                           topic_full=topic_full,
                           sign_data=sign_data)

@app.route("/sign/set", methods=["POST"])
def sign_set():
    sign = request.form.get("sign", "").strip()
    if sign:
        rep_idx = count_repetitions(state.topic, state.signer_id, sign)
        state.set(sign=sign, rep_idx=rep_idx, last_result=None, error=None)
        return redirect(url_for("record_page"))
    return redirect(url_for("sign_page"))

@app.route("/sign/add", methods=["POST"])
def sign_add():
    new = request.form.get("new_sign", "").strip()
    if new and state.topic:
        add_sign(state.topic, new)
    return redirect(url_for("sign_page"))

# — Record ——————————————————————————————————————————————————————————————————

@app.route("/record")
def record_page():
    if not state.sign:
        return redirect(url_for("sign_page"))
    signs = load_sign_list(state.topic)
    sign_full = next(
        (s for s in signs if s["uz"] == state.sign),
        {"uz": state.sign, "ru": state.sign, "en": state.sign},
    )
    return render_template("record.html",
                           signer_id=state.signer_id,
                           topic=state.topic,
                           sign=sign_full,
                           rep_idx=state.rep_idx,
                           countdown_secs=state.countdown_secs)

@app.route("/record/start", methods=["POST"])
def record_start():
    if state.recording or state.awaiting_decision or state.cmd:
        return jsonify({"status": "busy"})
    state.set(cmd="record", last_result=None, error=None)
    return jsonify({"status": "ok"})

@app.route("/record/stop", methods=["POST"])
def record_stop():
    if state.recording and time.time() >= state.countdown_end:
        state.set(stop_requested=True)
        return jsonify({"status": "ok"})
    return jsonify({"status": "not_recording"})

@app.route("/record/status")
def record_status():
    return jsonify(state.status_dict())

@app.route("/record/save", methods=["POST"])
def record_save():
    if state.awaiting_decision:
        state.set(decision="save")
    return jsonify({"status": "ok"})

@app.route("/record/discard", methods=["POST"])
def record_discard():
    if state.awaiting_decision:
        state.set(decision="discard")
    return jsonify({"status": "ok"})

@app.route("/record/set_countdown", methods=["POST"])
def record_set_countdown():
    secs = int(request.get_json(silent=True).get("secs", 3))
    if secs in (3, 4, 5) and not state.recording:
        state.set(countdown_secs=secs)
    return jsonify({"status": "ok", "countdown_secs": state.countdown_secs})

@app.route("/record/again", methods=["POST"])
def record_again():
    state.set(last_result=None)
    return redirect(url_for("record_page"))

@app.route("/record/done", methods=["POST"])
def record_done():
    state.set(sign=None, last_result=None)
    return redirect(url_for("sign_page"))

@app.route("/record/video_list")
def record_video_list():
    if not (state.topic and state.signer_id and state.sign):
        return jsonify([])
    vid_dir = path_videos(state.topic, state.signer_id, state.sign)
    if not vid_dir.exists():
        return jsonify([])
    reps = sorted(
        [d.name for d in vid_dir.iterdir()
         if d.is_dir() and re.fullmatch(r"rep-\d+", d.name) and (d / "video.mp4").exists()],
        key=lambda x: int(x.split("-")[1]),
    )
    return jsonify(reps)

@app.route("/record/video/<rep_name>")
def record_video(rep_name):
    if not re.fullmatch(r"rep-\d+", rep_name):
        return Response(status=400)
    if not (state.topic and state.signer_id and state.sign):
        return Response(status=404)
    video_path = path_videos(state.topic, state.signer_id, state.sign) / rep_name / "video.mp4"
    if not video_path.exists():
        return Response(status=404)
    return send_file(video_path, mimetype="video/mp4", conditional=True)

# — Camera ——————————————————————————————————————————————————————————————————

@app.route("/video_feed")
def video_feed():
    resp = Response(_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp

def _stream():
    last_seq = -1
    while True:
        frame, last_seq = frame_buf.get_next(last_seq)
        if frame is None:
            continue
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            continue
        data = jpeg.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(data)).encode() + b"\r\n"
            b"\r\n" +
            data +
            b"\r\n"
        )

@app.route("/camera/set", methods=["POST"])
def camera_set():
    data = request.get_json(silent=True) or {}
    idx  = data.get("index")
    if idx is not None and not state.recording and not state.awaiting_decision:
        state.set(pending_camera=int(idx))
    return jsonify({"status": "ok"})

@app.route("/camera/scan")
def camera_scan():
    cams = scan_cameras()
    state.set(cameras=cams)
    return jsonify(cams)

@app.route("/camera/ping")
def camera_ping():
    return jsonify({"alive": frame_buf.is_alive(), "camera_idx": state.camera_idx})

@app.route("/camera/test")
def camera_test():
    cam_name = next(
        (c["name"] for c in state.cameras if c["index"] == state.camera_idx),
        f"Camera {state.camera_idx}"
    )
    return render_template("camera_test.html", cam_name=cam_name)


# ── Camera / recording loop (background thread) ───────────────────────────────

def camera_loop(cap_holder: list) -> None:
    video_writer = None
    rep_dir: Path | None = None
    f_idx = 0

    while state.running:

        # ── Resolve pending decision from browser ──────────────────────────
        if state.awaiting_decision and state.decision and rep_dir is not None:
            _apply_decision(rep_dir, save=(state.decision == "save"))
            rep_dir = None
            video_writer = None

        # ── Camera switch ──────────────────────────────────────────────────
        if state.pending_camera is not None:
            new_cap = cv2.VideoCapture(state.pending_camera)
            if new_cap.isOpened():
                new_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
                new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                new_cap.set(cv2.CAP_PROP_FPS, FPS)
                cap_holder[0].release()
                cap_holder[0] = new_cap
                state.set(camera_idx=state.pending_camera, pending_camera=None)
            else:
                state.set(pending_camera=None)

        cap = cap_holder[0]
        ret, raw = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        raw     = cv2.flip(raw, 1)
        display = raw.copy()
        h, w    = display.shape[:2]

        # ── Start recording ────────────────────────────────────────────────
        if state.cmd == "record" and video_writer is None:
            state.set(cmd=None, recording=True,
                      countdown_end=time.time() + state.countdown_secs,
                      frame_count=0)
            rep_dir = path_videos(state.topic, state.signer_id, state.sign) / f"rep-{state.rep_idx}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(rep_dir / "video.mp4"), fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT)
            )
            f_idx = 0

        # ── Countdown overlay ──────────────────────────────────────────────
        if state.recording and time.time() < state.countdown_end:
            secs = int(state.countdown_end - time.time()) + 1
            cv2.putText(display, "Get ready!",
                        (w // 2 - 130, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            cv2.putText(display, str(secs),
                        (w // 2 - 60, h // 2 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 255), 10)

        # ── Recording overlay ──────────────────────────────────────────────
        elif state.recording:
            if state.stop_requested:
                if video_writer:
                    video_writer.release()
                    video_writer = None
                state.set(recording=False, stop_requested=False, awaiting_decision=True)
            else:
                if video_writer:
                    video_writer.write(raw)
                f_idx += 1
                state.set(frame_count=f_idx)
                cv2.circle(display, (30, 30), 12, (0, 0,220), -1)
                cv2.putText(display, f"  REC  {f_idx} frames",
                            (45, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                if state.sign:
                    cv2.putText(display, state.sign,
                                (30, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

        # ── Awaiting decision overlay ──────────────────────────────────────
        elif state.awaiting_decision:
            ov = display.copy()
            cv2.rectangle(ov, (0, h - 75), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.65, display, 0.35, 0, display)
            cv2.putText(display, "How was that take?",
                        (25, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
            cv2.putText(display, "SAVE: Enter        DISCARD: Delete",
                        (25, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        # ── Idle preview ───────────────────────────────────────────────────
        elif state.sign:
            cv2.putText(display, state.sign,
                        (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(display, f"rep {state.rep_idx + 1}  ·  {state.topic}",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (130, 130, 130), 1)
        else:
            cv2.putText(display, "Select a sign in the browser",
                        (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (110, 110, 110), 2)

        frame_buf.put(display)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    cams = scan_cameras()
    state.set(cameras=cams, camera_idx=cams[0]["index"] if cams else 0)

    cap = cv2.VideoCapture(state.camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print("Cannot open camera. Check VIDEO_DEVICE in app/mod01_config.py")
        return

    cap_holder = [cap]

    cam_t = threading.Thread(target=camera_loop, args=(cap_holder,), daemon=True)
    cam_t.start()

    time.sleep(0.6)
    webbrowser.open("http://127.0.0.1:5000")
    print("Web UI → http://127.0.0.1:5000   (Ctrl+C to quit)")

    try:
        # Flask runs on the main thread — no CV2 window needed
        app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)
    finally:
        state.set(running=False)
        time.sleep(0.2)
        cap_holder[0].release()


if __name__ == "__main__":
    main()
