import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)  # Use 0 or 1
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=1)  # Detect only one hand
colorR = (255, 0, 255)

cx, cy, w, h = 100, 100, 200, 200


class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # If the index finger tip is in the rectangle region
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
                cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor


rectList = []
for x in range(5):
    rectList.append(DragRect([x * 250 + 150, 150]))

# Create a resizable window and set its size
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 640, 360)  # Set the window size to 640x360

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera")
        break

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)  # Detect hands in the image

    if hands:
        # Get the first hand detected
        hand = hands[0]
        lmList = hand["lmList"]  # List of 21 landmark points

        # Extract (x, y) coordinates of index finger tip (landmark 8) and middle finger tip (landmark 12)
        index_finger_tip = lmList[8][:2]  # (x, y)
        middle_finger_tip = lmList[12][:2]  # (x, y)

        # Check the distance between index finger tip and middle finger tip
        l, _, _ = detector.findDistance(index_finger_tip, middle_finger_tip, img)

        if l < 30:  # If the distance is less than 30, consider it a pinch gesture
            for rect in rectList:
                rect.update(index_finger_tip)  # Move the rectangle to the index finger tip position

    ## Draw Transparency
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                      (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out = cv2.addWeighted(img, 1 - alpha, imgNew, alpha, 0)

    # Resize the output image to make the window smaller
    out = cv2.resize(out, (0, 0), fx=0.5, fy=0.5)  # Resize to 50% of the original size

    cv2.imshow("Image", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()