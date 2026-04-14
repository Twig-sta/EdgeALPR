# This script tests the ALPR pipeline by processing a sample image and displaying the results. 
# It uses OpenCV to read the image, process it through the ALPR pipeline, and visualize the detections. 
# The script prints the detection results to the console and shows an annotated image with detected license plates highlighted.
import cv2
import sys
import os

# Add the parent directory to the system path to allow imports from the alpr package
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