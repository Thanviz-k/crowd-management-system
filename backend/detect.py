import cv2
import threading
from ultralytics import YOLO
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

zone_data = {"A": 0, "B": 0, "C": 0, "D": 0}
SCALE = 30

def get_status(count):
    if count > 400:
        return "DANGER", (0, 0, 255)
    elif count > 200:
        return "MODERATE", (0, 165, 255)
    else:
        return "SAFE", (0, 255, 0)

@app.route("/zones")
def get_zones():
    return jsonify(zone_data)

def run_detection():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture("../videos/crowd.mp4")
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        if frame_count % 2 != 0:
            continue

        frame = cv2.resize(frame, (800, 500))
        h, w = frame.shape[:2]

        results = model(frame)
        counts = {"A": 0, "B": 0, "C": 0, "D": 0}

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if cx < w // 2 and cy < h // 2:
                        counts["A"] += SCALE
                    elif cx >= w // 2 and cy < h // 2:
                        counts["B"] += SCALE
                    elif cx < w // 2 and cy >= h // 2:
                        counts["C"] += SCALE
                    else:
                        counts["D"] += SCALE
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        zone_data.update(counts)

        for zone, (rx, ry) in zip(
            ["A", "B", "C", "D"],
            [(0, 0), (w // 2, 0), (0, h // 2), (w // 2, h // 2)]
        ):
            count = counts[zone]
            status, color = get_status(count)
            overlay = frame.copy()
            cv2.rectangle(overlay, (rx, ry), (rx + w // 2, ry + h // 2), color, -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.putText(frame, f"Zone {zone}: {count} ({status})",
                        (rx + 10, ry + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Start detection in background thread
threading.Thread(target=run_detection, daemon=True).start()

# Flask runs on main thread
print("Starting Flask on http://127.0.0.1:5000")
app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)