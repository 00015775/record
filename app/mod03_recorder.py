import time
from pathlib import Path

import cv2
from mod01_config import COUNTDOWN_SECONDS, FPS, FRAME_HEIGHT, FRAME_WIDTH
from mod02_storage import ensure_folders, path_videos


def record_one_repetition(cap, topic: str, signer_id: str, sign: str, rep_idx: int) -> Path:
    """Record one repetition and return the rep directory.

    The caller is responsible for deciding whether to keep (save) or delete
    the rep_dir — this function only records; it does not finalize.
    """
    ensure_folders(topic, signer_id, sign)

    rep_dir = path_videos(topic, signer_id, sign) / f"rep-{rep_idx}"
    rep_dir.mkdir(exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(rep_dir / "video.mp4"), fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    # countdown
    start_time = time.time()
    elapsed = 0
    while elapsed < COUNTDOWN_SECONDS:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Camera failed during countdown")
        frame = cv2.flip(frame, 1)
        secs_left = int(COUNTDOWN_SECONDS - elapsed) + 1
        cv2.putText(frame, f"Start in {secs_left}", (400, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 6)
        cv2.putText(frame, f"{signer_id}  |  {sign}  |  rep {rep_idx + 1}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Recorder", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            out.release()
            raise KeyboardInterrupt("User abort during countdown")
        elapsed = time.time() - start_time

    # record until signer presses 's'
    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Camera lost during recording")

        out.write(frame)
        f_idx += 1

        display = cv2.flip(frame, 1)
        cv2.putText(display, f"REC  Frame {f_idx}  |  press 's' to stop",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Recorder", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            break
        if key == ord("q"):
            out.release()
            raise KeyboardInterrupt("User abort")

    out.release()
    return rep_dir
