import cv2
import numpy as np
import os
import HandTrackingModule as htm
from flask import Blueprint, render_template
from tensorflow.keras.models import load_model
import keyboard
import pygame
import time


VirtualPainter = Blueprint("HandTrackingModule", __name__, static_folder="static",template_folder="templates")

@VirtualPainter.route("/feature")
def strt():
    ############## Color Attributes ###############
    WHITE = (255, 255, 255)
    BLACK = (0,0,0)
    RED = (0,0,255)
    YELLOW = (0,255,255)
    GREEN = (0,255,0)
    BACKGROUND = (255,255,255)
    FORGROUND = (0,255,0)
    BORDER = (0,255,0)
    lastdrawColor = (0,0,1)
    drawColor = (0,0,255)
    BOUNDRYINC = 5

    ############## CV2 Attributes ###############
    cap = cv2.VideoCapture(0)
    width, height = 1280, 720
    cap.set(3, width)          #640, 1280
    cap.set(4, height)         #480, 720
    imgCanvas = np.zeros((height,width,3), np.uint8)


    ############## PyGame Attributes ###############
    pygame.init()
    FONT = pygame.font.SysFont('freesansbold.tff', 18)
    DISPLAYSURF = pygame.display.set_mode((width, height),flags=pygame.HIDDEN)
    pygame.display.set_caption("Digit Board")
    number_xcord = []
    number_ycord = []

    ############## Header Files Attributes ###############
    folderPath = "header"
    myList = os.listdir(folderPath)
    overlayList = []

    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)
    header = overlayList[0]


    ############## HandDetection Attributes ###############
    detector = htm.handDetector(detectionCon=0.85)
    xp , yp = 0, 0
    brushThickness = 15
    eraserThickness = 30
    modeValue = "OFF"
    modeColor = WHITE

    while True:
        SUCCESS, img = cap.read()
        img = cv2.flip(img,1)

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=True)

        if len(lmList)>0:

            
            fingers = detector.fingersUp()
            pfx,pfy = lmList[detector.tipIds[1]][1:]


            # End program when pointer goes to exit button
            if pfy < 125 and 1160 < pfx < 1250:
                cap.release()
                cv2.destroyAllWindows()
                return render_template("index.html")
                quit()

            # Drawing Mode - Index finger up
            if fingers[1] and fingers[2] == False:

                cv2.circle(img, (pfx,pfy-15), 15, drawColor, cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = pfx, pfy

                if drawColor == (0, 0, 0):
                    cv2.line(img, (xp,yp), (pfx,pfy), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp,yp), (pfx,pfy), drawColor, eraserThickness)
                else:
                    cv2.line(img, (xp,yp), (pfx,pfy), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp,yp), (pfx,pfy), drawColor, brushThickness)
                    pygame.draw.line(DISPLAYSURF, WHITE, (xp,yp), (pfx,pfy), brushThickness)
                xp, yp = pfx, pfy
            else:
                xp, yp = 0, 0

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)
        
        img[0:132,0:1280] = header
        pygame.display.update()
        # cv2.imshow("Paint",imgCanvas)
        cv2.imshow("Image",img)
        cv2.waitKey(1)

strt()