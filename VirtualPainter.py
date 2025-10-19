import cv2
import numpy as np
import os
import HandTrackingModule as htm
from flask import Blueprint, render_template
from tensorflow.keras.models import load_model
import keyboard
import pygame
import time
import Ghost


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
    shape_xcoords = []
    shape_ycoords = []


    ############## Asset Files Attributes ###############
    folderPath = "assets"

    ghostimg = cv2.imread(os.path.join(folderPath, "ghost.png"), cv2.IMREAD_UNCHANGED)
    ghostHeight, ghostWidth = 150, 150
    ghostimg = cv2.resize(ghostimg, (ghostWidth, ghostHeight))


    wandimg = cv2.imread(os.path.join(folderPath, "wand.png"))
    wandHeight, wandWidth, _ = wandimg.shape


    ############## Predication Model Attributes ###############
    label=""
    model = load_model("gdraw.keras")
    shapeLABELS = { 0: "down", 1: "horz", 2: "up", 3: "vert"}

    rect_min_x, rect_min_y = 0,0
    rect_max_x, rect_max_y = 0,0

    ############## HandDetection Attributes ###############
    detector = htm.handDetector(detectionCon=0.85)
    drawMode = "OFF"
    xp , yp = 0, 0
    brushThickness = 15

    ############## Game Attributes ###############
    playerScore = 0
    playerHbWidth, playerHbHeight = 100, 100
    playerPos = (width//2 - playerHbWidth//2, height*2//3 - playerHbHeight//2)
    ghosts = [ Ghost.Ghost(np.random.randint(0, width - ghostWidth), np.random.randint(0, height - ghostHeight), playerPos[0] + playerHbWidth//2, playerPos[1] + playerHbHeight//2, speed=5) for _ in range(5) ]

    ############## Main Loop ###############
    while True:
        SUCCESS, img = cap.read()
        img = cv2.flip(img,1)

        ########## DRAWING INTERFACE ##########
        cv2.putText(img,"Press SPACE for draw mode",(0,145),3,0.5,BLACK,1,cv2.LINE_AA)
        cv2.putText(img,f'{"DRAW MODE IS "}{drawMode}',(0,162),3,0.5,BLACK,1,cv2.LINE_AA)

        if keyboard.is_pressed('space'):
            if drawMode == "OFF":
                drawMode = "ON"
            else:
                drawMode = "OFF"
                imgCanvas = np.zeros((height,width,3), np.uint8)
                
            time.sleep(0.3)

        img = detector.findHands(img, draw = False)
        lmList = detector.findPosition(img, draw = True)
        
        # Hand Detected
        if len(lmList)>0:

            fist = detector.fistOrientation()

            # Top pointer knuckle
            topknuckle = 5
            # Bottom pointer knuckle
            botknuckle = 6

            # Offset distance   
            offset = 100
            # Draw from knuckles of pointer finger with offset
            drawx,drawy = (lmList[topknuckle][1] + lmList[botknuckle][1]) // 2, (lmList[topknuckle][2] + lmList[botknuckle][2]) // 2 - offset

            if drawMode == "ON":
                # Predict mode
                if fist == "horizontal":
                    drawColor = BLACK

                    shape_xcoords = sorted(shape_xcoords)
                    shape_ycoords = sorted(shape_ycoords)

                    if(len(shape_xcoords) > 0 and len(shape_ycoords)>0):
                        
                        # Draw rectangle around the drawn shape
                        pad = 50
                        rect_min_x, rect_max_x = max(shape_xcoords[0]-BOUNDRYINC - pad, 0), min(width, shape_xcoords[-1]+BOUNDRYINC + pad)
                        rect_min_y, rect_max_y = max(0, shape_ycoords[0]-BOUNDRYINC - pad), min(shape_ycoords[-1]+BOUNDRYINC + pad, height)
                        cv2.rectangle(imgCanvas,(rect_min_x,rect_min_y),(rect_max_x,rect_max_y),BROWN,3)

                        img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x,rect_min_y:rect_max_y].T.astype(np.float32) 

                        # Normalize and resize the image before predication
                        image = cv2.resize(img_arr, (50,50))
                        image = cv2.resize(image,(50,50))/255

                        # Get label from model prediction
                        label = str(shapeLABELS[np.argmax(model.predict(image.reshape(1,50,50,1)))])

                        # Display the label on the screen
                        cv2.rectangle(imgCanvas,(rect_min_x+50,rect_min_y-20),(rect_min_x,rect_min_y),WHITE,-1)
                        cv2.putText(imgCanvas,label,(rect_min_x,rect_min_y-5),3,0.5,GREEN,1,cv2.LINE_AA)

                        # Reset coordinates for next shape
                        shape_xcoords = []
                        shape_ycoords = []

                    xp, yp = 0, 0

                # Drawing Mode
                elif fist == "vertical":
                    drawColor = GREEN

                    shape_xcoords.append(drawx)
                    shape_ycoords.append(drawy)
                    
                    if xp == 0 and yp == 0:
                        xp, yp = drawx, drawy

                    cv2.circle(img, (drawx,drawy-15), 15, drawColor, cv2.FILLED)

                    cv2.line(img, (xp,yp), (drawx,drawy), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp,yp), (drawx,drawy), drawColor, brushThickness)
                    pygame.draw.line(DISPLAYSURF, WHITE, (xp,yp), (drawx,drawy), brushThickness)
                    xp, yp = drawx, drawy

                else:
                    xp, yp = 0, 0

            # End program when pointer goes to exit button
            if drawx > 1160 and drawy < 125:
                cap.release()
                cv2.destroyAllWindows()
                return render_template("index.html")
                quit()

        ########## GAME ##########

        cv2.putText(img,f'Score: {playerScore}',(10,70),3,1,BLACK,2,cv2.LINE_AA)
        
        # Draw player hitbox
        cv2.rectangle(img, playerPos, (playerPos[0]+playerHbWidth, playerPos[1]+playerHbHeight), RED, 2)

        # Move and draw ghosts
        for g in ghosts:
            g.move()
            ghostX, ghostY = int(g.x), int(g.y)
            overlay_transparent(img, ghostimg, ghostX, ghostY)

            # Check collision with player
            if (ghostX + ghostWidth > playerPos[0] and ghostX < playerPos[0] + playerHbWidth and
                ghostY + ghostHeight > playerPos[1] and ghostY < playerPos[1] + playerHbHeight):
                playerScore += 1
                # Reset ghost position
                g.x = np.random.randint(0, width - ghostWidth)
                g.y = np.random.randint(0, height - ghostHeight)
            


        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)
        
        #img[0:125, 1160:1280] = header
        pygame.display.update()
        cv2.imshow("Image",img)
        cv2.waitKey(1)

def overlay_transparent(bg, fg, x, y):
    h, w = fg.shape[:2]

    # Split the channels of the foreground
    b, g, r, a = cv2.split(fg)
    alpha = a.astype(float) / 255.0
    alpha = cv2.merge([alpha, alpha, alpha])

    # Region of interest (ROI) in the background
    roi = bg[y:y+h, x:x+w]

    # Convert fg to BGR only
    fg_rgb = cv2.merge([b, g, r]).astype(float)
    bg_rgb = roi.astype(float)

    # Blend the images
    blended = cv2.add(fg_rgb * alpha, bg_rgb * (1 - alpha))

    bg[y:y+h, x:x+w] = blended.astype(np.uint8)

strt()