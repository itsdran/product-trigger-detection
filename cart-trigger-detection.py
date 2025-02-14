import cv2
import numpy as np
from skimage.filters import threshold_otsu

def detect_line(frame, stored_line_y):
    """Detect highest horizontal line in frame using edge detection"""
    if stored_line_y is not None:
        return None, stored_line_y  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    height = frame.shape[0]
    roi = edges[:height//2, :]

    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 100, minLineLength=200, maxLineGap=20)

    if lines is not None:
        min_y = height
        best_line = None

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 100 and y1 < min_y:
                min_y = y1
                best_line = (x1, y1, x2, y2)

        return best_line, min_y

    return None, None

def detect_crossing_objects(frame, horizontal_line):
    """Detect objects that cross the horizontal line"""
    if horizontal_line is None:
        return [], np.zeros(frame.shape[:2], dtype=np.uint8)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh_value = threshold_otsu(blurred)
    _, binary = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)

    binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    object_mask = np.zeros_like(gray)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  
            x, y, w, h = cv2.boundingRect(contour)

            object_bottom = y + h  
            object_top = y  

            # Trigger only when object crosses the horizontal line
            if object_top < horizontal_line < object_bottom:
                valid_contours.append((x, y, w, h))
                cv2.drawContours(object_mask, [contour], 0, 255, -1)

    return valid_contours, object_mask

cap = cv2.VideoCapture(0)
stored_line_y = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    line, line_y = detect_line(frame, stored_line_y)
    if line_y is not None:
        stored_line_y = line_y  

    if stored_line_y is not None:
        # Draw the detected line (Green)
        cv2.line(display_frame, (0, stored_line_y), (frame.shape[1], stored_line_y), (0, 255, 0), 2)

    # Detect objects crossing the horizontal line
    objects, mask = detect_crossing_objects(frame, stored_line_y)

    # Second view: Completely black until an object crosses
    segmented = np.zeros_like(frame)  

    if len(objects) > 0:
        segmented = cv2.bitwise_and(frame, frame, mask=mask)

    for x, y, w, h in objects:
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display_frame, "Product Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    combined = np.hstack((display_frame, segmented))

    cv2.imshow('Restricted Detection', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
