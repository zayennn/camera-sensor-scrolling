import cv2
import mediapipe as mp
import time
import pyautogui

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

pTime = 0

def is_finger_vertical_up(lms, tip_id, pip_id):
    tip = lms[tip_id]
    pip = lms[pip_id]
    return tip.y < pip.y and abs(tip.x - pip.x) < 0.03

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Pastikan ada deteksi tangan dan handedness info
    if results.multi_hand_landmarks and results.multi_handedness:
        for i, handLms in enumerate(results.multi_hand_landmarks):
            # Ambil jenis tangan (Right/Left)
            handType = results.multi_handedness[i].classification[0].label

            if handType == "Right":
                lmList = handLms.landmark

                index_up = is_finger_vertical_up(lmList, 8, 6)
                middle_up = is_finger_vertical_up(lmList, 12, 10)

                if index_up and not middle_up:
                    print("Scroll down (RIGHT hand)")
                    pyautogui.scroll(-20)
                    time.sleep(1)

                elif index_up and middle_up:
                    print("Scroll up (RIGHT hand)")
                    pyautogui.scroll(20)
                    time.sleep(1)

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            else:
                print("Tangan kiri terdeteksi - abaikan")

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()