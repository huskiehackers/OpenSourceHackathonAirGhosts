import cv2
import mediapipe as mp

class handDetector():
    def __init__(self, mode = False, maxHands = 2, modelComplexity = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    if id == 0:
                        cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
                    elif id % 4 == 1 and id != 1:
                        cv2.circle(img, (cx,cy), 5, (0,0,255), cv2.FILLED)
                    elif id % 4 == 2 and id != 2:
                        cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    
    def fistClosed(self):
        fingers = self.fingersUp()
        if fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            return True
        else:
            return False

    def fistOrientation(self):
        knuckles = [self.lmList[5], self.lmList[9], self.lmList[13], self.lmList[17]]

        # Calculate average x and y positions of knuckles
        x_positions = [pt[1] for pt in knuckles]
        y_positions = [pt[2] for pt in knuckles]

        x_range = max(x_positions) - min(x_positions)
        y_range = max(y_positions) - min(y_positions)

        tolerance = 20  # pixels, adjust as needed

        if x_range > y_range + tolerance:
            return "horizontal"
        elif y_range > x_range + tolerance:
            return "vertical"

