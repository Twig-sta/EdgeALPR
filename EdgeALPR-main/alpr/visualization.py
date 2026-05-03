import cv2

def draw_detections(frame, detections):
    for det in detections:
        x, y, w, h = det['bbox']
        text = det["text"]
        status = det.get("status", "unknown")

        if status == "authorized":
            color = (0, 255, 0)  # green
        else:
            color = (0, 0, 255)  # red

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.rectangle(frame, (x, y - 30), (x + w, y), color, -1)

        label = f"{text} ({status})"

        cv2.putText(
            frame,
            label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    return frame