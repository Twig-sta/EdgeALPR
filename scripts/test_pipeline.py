import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alpr.pipeline import process_frame
from alpr.visualization import draw_detections

image_path = "tests/images/BMW_license_plate.jpg"

frame = cv2.imread(image_path)

results = process_frame(frame)

print("Detection Results:")
print(results)

# Draw detections
annotated = draw_detections(frame, results)

cv2.imshow("Plate Detection", annotated)

cv2.waitKey(0)
cv2.destroyAllWindows()