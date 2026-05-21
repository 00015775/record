import cv2
import datetime
import os
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Measure actual FPS instead of trusting cap.get(CAP_PROP_FPS) which is wrong on macOS
print("Measuring camera FPS...")
for _ in range(5):
    cap.read()  # discard first few frames while camera warms up
t0 = time.time()
for _ in range(30):
    cap.read()
fps = 30 / (time.time() - t0)
print(f"Detected FPS: {fps:.1f}  |  Resolution: {width}x{height}")

writer = None
recording = False

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(data_dir, exist_ok=True)

print("Press R to start/stop recording. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if recording and writer:
        writer.write(frame)
        cv2.putText(frame, "REC", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("C922 Recorder", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("r"):
        if not recording:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(data_dir, f"recording_{timestamp}.mp4")
            writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            recording = True
            print(f"Recording started: {filename}")
        else:
            recording = False
            writer.release()
            writer = None
            print("Recording saved.")

if recording and writer:
    writer.release()
cap.release()
cv2.destroyAllWindows()
