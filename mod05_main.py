import cv2
from mod01_config import VIDEO_DEVICE, FRAME_WIDTH, FRAME_HEIGHT, FPS
from mod04_ui import select_signer, select_topic, select_sign, after_recording_menu
from mod03_recorder import record_one_repetition
from mod02_storage import count_repetitions


def main():
    cap = cv2.VideoCapture(VIDEO_DEVICE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print("Cannot open camera. Check VIDEO_DEVICE in mod01_config.py")
        return

    signer_id = select_signer()

    while True:
        # ── topic ──────────────────────────────────────────────────────────────
        topic = select_topic(signer_id)
        if topic is None:
            signer_id = select_signer()
            continue

        # ── sign word ──────────────────────────────────────────────────────────
        while True:
            chosen_sign = select_sign(signer_id, topic)
            if chosen_sign is None:
                break  # back to topic menu

            rep_idx = count_repetitions(topic, signer_id, chosen_sign)

            # ── recording loop for this sign ───────────────────────────────────
            while True:
                # wait for 's' to start
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, "Press 's' to start", (300, 360),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                    cv2.putText(
                        frame,
                        f"{signer_id} | {topic} | {chosen_sign} | rep {rep_idx + 1}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                    )
                    cv2.imshow("Recorder", frame)
                    k = cv2.waitKey(10) & 0xFF
                    if k == ord('s'):
                        break
                    if k == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                try:
                    rep_idx = record_one_repetition(cap, topic, signer_id, chosen_sign, rep_idx)
                except KeyboardInterrupt:
                    print("\nRecording aborted.")
                    break
                except Exception as e:
                    print(f"\nError during recording: {e}")
                    break

                action = after_recording_menu(signer_id, topic, chosen_sign, rep_idx - 1)
                if action == "done":
                    break  # back to sign list

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
