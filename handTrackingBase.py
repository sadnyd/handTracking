import warnings
warnings.filterwarnings('ignore')

import cv2
import mediapipe as mp
import time




cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils


pTime = 0
cTime = 0


while True:
    success, img = cap.read()
    imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results = hands.process(imageRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for idLm, lm in enumerate(handLms.landmark):
                # print(idLm,lm)
                h, w, c =  img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(idLm, cx,cy)
                if idLm == 9:
                    cv2.circle(img, (cx, cy), 10,(0,128,0),cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    if not success:
        print("Failed to capture image")
        break

    if img is None or img.size == 0:
        print("Captured empty frame")
        continue


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,80),cv2.FONT_ITALIC,2,
                (0,255,255),3)
    cv2.imshow("Image", img)


    # print("key code:", cv2.waitKey(1) if cv2.waitKeyEx(1)!=-1 else 0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
