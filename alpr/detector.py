#**This file detects license plate candidates in the preprocessed image**#

#import necessary libraries cv2 for image processing and the utility functions for preprocessing and edge detection
import cv2 
from alpr.utils import preprocess_image, detect_edges 

#This function detects potential license plate regions in the input frame
def detect_plates(frame):
    #Preprocess the image and detect edges to find contours that may correspond to license plates
    processed = preprocess_image(frame)
    edges = detect_edges(processed)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    debug = frame.copy()

    #Loop through the detected contours and filter them based on aspect ratio and area to identify potential license plate candidates
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(debug, (x, y), (x+w, y+h), (0,255,0), 1)
        aspect_ratio = w / float(h)
        area = w * h

        print(f"Candidate: w={w}, h={h}, area={area}, ratio={aspect_ratio:.2f}")
        

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        
        if (2.5 <= aspect_ratio <= 7.5 and 1000 <= area <= 60000 and h >= 20 and w >= 80):
            
            
            crop_y1 = int(y + h *0.3)
            crop_y2 = int(y + h *0.75)
            plate_img = frame[y:y+h, x:x+w]

            if plate_img is None or plate_img.size == 0:

                continue

            #Store the bounding box and the corresponding image of the candidate license plate for further processing
            candidates.append({
                'bbox': (x, y, w, h),
                'image': plate_img
            })
   
    return candidates