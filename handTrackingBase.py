import cv2
import mediapipe as mp
import time

import warnings
warnings.filterwarnings('ignore')


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()
    imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results = hands.process(imageRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw the landmarks on the original BGR image
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)



    if not success:
        print("Failed to capture image")
        break


    if img is None or img.size == 0:
        print("Captured empty frame")
        continue


    cv2.imshow("Image", img)
    # print("key code:", cv2.waitKeyEx(1) if cv2.waitKeyEx(1)!=-1 else 0)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
