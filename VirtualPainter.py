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
    BLUE = (255,0,0)
    BROWN = (19,69,139)
    GREEN = (0,255,0)
    drawColor = GREEN
    BOUNDRYINC = 5

    ############## CV2 Attributes ###############
    cap = cv2.VideoCapture(0)
    width, height = 1280, 720
    cap.set(3, width)          #640, 1280
    cap.set(4, height)         #480, 720
    imgCanvas = np.zeros((height,width,3), np.uint8)


    ############## PyGame Attributes ###############
    pygame.init()
    DISPLAYSURF = pygame.display.set_mode((width, height),flags=pygame.HIDDEN)
    pygame.display.set_caption("Digit Board")
    number_xcord = []
    number_ycord = []


    ############## Header Files Attributes ###############
    
    '''
    folderPath = "header"
    myList = os.listdir(folderPath)
    overlayList = []

    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)
    header = overlayList[0]
    '''

    ############## Predication Model Attributes ###############
    label=""
    model = load_model("gdraw.keras")
    shapeLABELS = { 0: "down", 1: "horz", 2: "up", 3: "vert"}

    rect_min_x, rect_min_y = 0,0
    rect_max_x, rect_max_y = 0,0

    ############## HandDetection Attributes ###############
    detector = htm.handDetector(detectionCon=0.85)
    xp , yp = 0, 0
    brushThickness = 15
    eraserThickness = 30

    while True:
        SUCCESS, img = cap.read()
        img = cv2.flip(img,1)

        img = detector.findHands(img, draw = False)
        lmList = detector.findPosition(img, draw = True)
        
        # Hand Detected
        if len(lmList)>0:

            fist = detector.fistOrientation()
            fingers = detector.fingersUp()
            pfx,pfy = lmList[detector.tipIds[1]][1:]

            # Detect mode
            if fist == "horizontal":
                drawColor = BLACK

                number_xcord = sorted(number_xcord)
                number_ycord = sorted(number_ycord)

                if(len(number_xcord) > 0 and len(number_ycord)>0):
                    rect_min_x, rect_max_x = max(number_xcord[0]-BOUNDRYINC, 0)-50, min(width, number_xcord[-1]+BOUNDRYINC)+50
                    rect_min_y, rect_max_y = max(0, number_ycord[0]-BOUNDRYINC)-50, min(number_ycord[-1]+BOUNDRYINC, height)+50
                    number_xcord = []
                    number_ycord = []

                    img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x,rect_min_y:rect_max_y].T.astype(np.float32) 
                    cv2.rectangle(imgCanvas,(rect_min_x,rect_min_y),(rect_max_x,rect_max_y),BROWN,3)
                    image = cv2.resize(img_arr, (50,50))
                    image = np.pad(image, (10,10), 'constant' , constant_values =0)
                    image = cv2.resize(image,(50,50))/255
                    label = str(shapeLABELS[np.argmax(model.predict(image.reshape(1,50,50,1)))])

                    pygame.draw.rect(DISPLAYSURF,BLACK,(0,0,width,height))

                    cv2.rectangle(imgCanvas,(rect_min_x+50,rect_min_y-20),(rect_min_x,rect_min_y),WHITE,-1)
                    cv2.putText(imgCanvas,label,(rect_min_x,rect_min_y-5),3,0.5,GREEN,1,cv2.LINE_AA)

                pygame.draw.line(DISPLAYSURF, BLACK, (xp,yp), (pfx,pfy), eraserThickness)

            # End program when pointer goes to exit button
            if pfy < 125 and 1160 < pfx < 1250:
                cap.release()
                cv2.destroyAllWindows()
                return render_template("index.html")
                quit()

            #Drawing Mode
            elif fist == "vertical":

                number_xcord.append(pfx)
                number_ycord.append(pfy)
                
                if xp == 0 and yp == 0:
                    xp, yp = pfx, pfy

                drawColor = GREEN
                cv2.circle(img, (pfx,pfy-15), 15, drawColor, cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = pfx, pfy

                cv2.line(img, (xp,yp), (pfx,pfy), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp,yp), (pfx,pfy), drawColor, brushThickness)
                pygame.draw.line(DISPLAYSURF, WHITE, (xp,yp), (pfx,pfy), brushThickness)
                xp, yp = pfx, pfy


                xp, yp = pfx, pfy

            #elif fingers[1] == 1 and fingers[2] == 1:
                # Erase Whole Canvas
                #imgCanvas = np.zeros((height, width, 3), np.uint8)  # Clears OpenCV canvas
                #DISPLAYSURF.fill(BLACK)  # Clears PyGame surface


            else:
                xp, yp = 0, 0

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)
        
        #img[0:132,0:1280] = header
        pygame.display.update()
        cv2.imshow("Image",img)
        cv2.waitKey(1)

strt()