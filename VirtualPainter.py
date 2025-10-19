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
import random


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
    MAGENTA = (255,0,255)
    drawColor = MAGENTA
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

    heartWidth, heartHeight = 200, 200
    heartimg = cv2.imread(os.path.join(folderPath, "heart.png"), cv2.IMREAD_UNCHANGED)
    heartimg = cv2.resize(heartimg, (heartWidth, heartHeight))

    ghostHeight, ghostWidth = 150, 150
    ghostimg = cv2.imread(os.path.join(folderPath, "ghost.png"), cv2.IMREAD_UNCHANGED)
    ghostimg = cv2.resize(ghostimg, (ghostWidth, ghostHeight))

    symbolSize = (40, 40)
    upimg = cv2.imread(os.path.join(folderPath, "UpArr.png"), cv2.IMREAD_UNCHANGED)
    upimg = cv2.resize(upimg, symbolSize)
    downimg = cv2.imread(os.path.join(folderPath, "DownArr.png"), cv2.IMREAD_UNCHANGED)
    downimg = cv2.resize(downimg, symbolSize)
    horzimg = cv2.imread(os.path.join(folderPath, "Hori.png"), cv2.IMREAD_UNCHANGED)
    horzimg = cv2.resize(horzimg, symbolSize)
    vertimg = cv2.imread(os.path.join(folderPath, "Vert.png"), cv2.IMREAD_UNCHANGED)
    vertimg = cv2.resize(vertimg, symbolSize)

    symbolImages = [downimg, horzimg, upimg, vertimg]
    symbolColors = [GREEN, YELLOW, RED, BLUE]

    wandWidth, wandHeight = 150, 150
    wandimg = cv2.imread(os.path.join(folderPath, "wand.png"), cv2.IMREAD_UNCHANGED)
    wandimg = cv2.resize(wandimg, (wandWidth, wandHeight))

    wandimg_LHanded = cv2.rotate(wandimg, cv2.ROTATE_90_CLOCKWISE)
    wandimg_RHanded = cv2.rotate(wandimg, cv2.ROTATE_90_COUNTERCLOCKWISE)

    wandHand = ""

    ############## Predication Model Attributes ###############
    label=""
    model = load_model("gdraw.keras")
    shapeLABELS = { 0: "down", 1: "horz", 2: "up", 3: "vert"}

    rect_min_x, rect_min_y = 0,0
    rect_max_x, rect_max_y = 0,0

    ############## HandDetection Attributes ###############
    detector = htm.handDetector(detectionCon=0.85)
    gameRunning = "OFF"
    xp , yp = 0, 0
    brushThickness = 15
    labelIdx = -1

    ############## Game Attributes ###############
    playerScore = 0
    maxLives = 5
    currentLives = maxLives
    playerHbWidth, playerHbHeight = 75, 75
    playerPos = (width//2 - playerHbWidth//2, height*2//3 - playerHbHeight//2)

    ghostSpeed = 2
    ghostSpawnX = (0 - ghostWidth, width + ghostWidth)
    ghostSpawnRate = 1
    ghostSpawnCap = 1
    ghosts = []

    ############## Main Loop ###############
    while True:
        SUCCESS, img = cap.read()
        img = cv2.flip(img,1)

        # Open Screen
        if gameRunning == "OFF":
            cv2.putText(img,"AIR DRAW: GHOST DRAW",(0, height // 4),3,3,WHITE,3,cv2.LINE_AA)
            cv2.putText(img,"PRESS SPACE TO START",(0, height // 2),3,2,WHITE,3,cv2.LINE_AA)
            cv2.putText(img,"PRESS ESCAPE TO EXIT",(0, height * 3 // 4),3,2,WHITE,3,cv2.LINE_AA)

            if keyboard.is_pressed('space'):
                gameRunning = "ON"
            if keyboard.is_pressed('escape'):
                cap.release()
                cv2.destroyAllWindows()
                return render_template("index.html")
                quit()

        # Player Dies
        elif gameRunning == "DEAD":
            cv2.putText(img,"YOU DIED",(width // 2 - 300, height // 4),3,3,RED,3,cv2.LINE_AA)
            cv2.putText(img,"PRESS SPACE TO START",(0, height // 2),3,2,RED,3,cv2.LINE_AA)
            cv2.putText(img,"PRESS ESCAPE TO EXIT",(0, height * 3 // 4),3,2,RED,3,cv2.LINE_AA)

            if keyboard.is_pressed('space'):
                # Reset game attributes
                playerScore = 0
                currentLives = maxLives
                ghostSpeed = 2
                ghostSpawnCap = 1
                ghosts = []
                gameRunning = "ON"

            if keyboard.is_pressed('escape'):
                cap.release()
                cv2.destroyAllWindows()
                return render_template("index.html")
                quit()
            
        ########## GAME RUNNING ##########
        else:
            ########## HAND DETECTION ##########
            img = detector.findHands(img, draw = False)
            lmList = detector.findPosition(img, draw = False)
            
            # Hand Detected
            if len(lmList)>0:

                fist = detector.fistOrientation()

                # Top pointer knuckle
                topknuckle = 5
                # Bottom pointer knuckle
                botknuckle = 6

                # Offset distance   
                offset = 150
                # Draw from knuckles of pointer finger with offset
                drawx,drawy = (lmList[topknuckle][1] + lmList[botknuckle][1]) // 2, (lmList[topknuckle][2] + lmList[botknuckle][2]) // 2 - offset

                # Predict mode
                if fist == "horizontal":

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
                        labelIdx = np.argmax(model.predict(image.reshape(1,50,50,1)))

                        # Check ghost label and remove dead ghosts
                        dead_ghosts = []
                        for g in ghosts:
                            g.checkMatch(labelIdx)
                            if g.isDead():
                                dead_ghosts.append(g)
                                playerScore += 1

                        for g in dead_ghosts:
                            ghosts.remove(g)

                        # Clear Screen
                        imgCanvas = np.zeros((height,width,3), np.uint8)

                        # Reset coordinates for next shape
                        shape_xcoords = []
                        shape_ycoords = []

                    if (wandHand == "L"):
                        overlay_transparent(img, wandimg_LHanded, drawx, drawy + wandHeight //2)
                        cv2.circle(img, (drawx + offset, drawy-30 + offset), 30, symbolColors[labelIdx], cv2.FILLED)
                    else:
                        overlay_transparent(img, wandimg_RHanded, drawx - wandWidth, drawy + wandHeight //2)
                        cv2.circle(img, (drawx - offset, drawy-30 + offset), 30, symbolColors[labelIdx], cv2.FILLED)

                    xp, yp = 0, 0

                # Drawing Mode
                elif fist == "vertical":

                    # Identify left or right hand
                    if (lmList[botknuckle][1] > lmList[topknuckle][1]):
                        wandHand = "L"
                    else:
                        wandHand = "R"

                    overlay_transparent(img, wandimg, drawx - wandWidth //2, drawy)

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
                    overlay_transparent(img, wandimg, drawx - wandWidth //2, drawy)
                    xp, yp = 0, 0

                # End program when pointer goes to exit button
                if drawx > 1160 and drawy < 125:
                    cap.release()
                    cv2.destroyAllWindows()
                    return render_template("index.html")
                    quit()

            ########## GAME ##########

            cv2.putText(img,f'Score: {playerScore}',(10,70),3,3,WHITE,3,cv2.LINE_AA)
            
            # Draw player heart
            overlay_transparent(img, heartimg, playerPos[0] - (heartWidth - playerHbWidth) // 2, playerPos[1] - (heartHeight - playerHbHeight) // 2)
            cv2.putText(img,f'{currentLives}',(playerPos[0] + playerHbWidth//2 - 30, playerPos[1] + playerHbHeight//2 + 30),3,3,WHITE,3,cv2.LINE_AA)

            # Create ghost
            if (random.randint(0,100) <= ghostSpawnRate and len(ghosts) < ghostSpawnCap):
                ghosts.append(Ghost.Ghost(ghostSpawnX[random.randint(0,1)], random.randint(0 - ghostHeight, height - ghostHeight), playerPos[0] + playerHbWidth//2, playerPos[1] + playerHbHeight//2, ghostSpeed))

                # Update ghost cap and speed
                match playerScore:
                    case 3:
                        ghostSpawnCap = 2
                        ghostSpeed = 3
                    case 8:
                        ghostSpawnCap = 4
                        ghostSpeed = 4
                    case 15:
                        ghostSpawnCap = 8
                        ghostSpeed = 5

            # Ghost loop
            dead_ghosts = []
            for g in ghosts:
                g.move()
                ghostX, ghostY = int(g.x), int(g.y)
                if (g.flipped):
                    ghostimg_flipped = cv2.flip(ghostimg, 1)
                    overlay_transparent(img, ghostimg_flipped, ghostX, ghostY)
                else:
                    overlay_transparent(img, ghostimg, ghostX, ghostY)

                symbolIdx = 0
                for s in g.symbols:
                    symbolX, symbolY = ghostX + (symbolIdx * symbolSize[0]) + 10, ghostY + ghostHeight + 10
                    overlay_transparent(img, symbolImages[s], symbolX, symbolY)
                    symbolIdx += 1

                # Check collision with player
                if (ghostX + ghostWidth > playerPos[0] and ghostX < playerPos[0] + playerHbWidth and
                    ghostY + ghostHeight > playerPos[1] and ghostY < playerPos[1] + playerHbHeight):
                        currentLives -= 1

                        if (currentLives == 0):
                            gameRunning = "DEAD"
                        
                        dead_ghosts.append(g)

            for g in dead_ghosts:
                ghosts.remove(g)


        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        pygame.display.update()
        cv2.imshow("Image",img)
        cv2.waitKey(1)

def overlay_transparent(bg, fg, x, y):
    bg_h, bg_w = bg.shape[:2]
    fg_h, fg_w = fg.shape[:2]

    # Completely outside (to the right or bottom)
    if x >= bg_w or y >= bg_h or x + fg_w <= 0 or y + fg_h <= 0:
        return

    # Crop foreground if partially outside the left or top
    fg_x_start = 0
    fg_y_start = 0

    if x < 0:
        fg_x_start = -x
        fg_w = fg_w + x  # since x is negative
        x = 0
    if y < 0:
        fg_y_start = -y
        fg_h = fg_h + y
        y = 0

    # Limit width and height to fit background
    overlay_w = min(fg_w, bg_w - x)
    overlay_h = min(fg_h, bg_h - y)

    # If after clipping it's invalid, skip
    if overlay_w <= 0 or overlay_h <= 0:
        return

    # Crop fg to match the overlay region
    fg_crop = fg[fg_y_start:fg_y_start+overlay_h, fg_x_start:fg_x_start+overlay_w]

    # Split the channels
    b, g, r, a = cv2.split(fg_crop)
    alpha = a.astype(float) / 255.0
    alpha = cv2.merge([alpha, alpha, alpha])

    # ROI in bg
    roi = bg[y:y+overlay_h, x:x+overlay_w]

    fg_rgb = cv2.merge([b, g, r]).astype(float)
    bg_rgb = roi.astype(float)

    # Blend
    blended = fg_rgb * alpha + bg_rgb * (1 - alpha)

    # Put blended result back
    bg[y:y+overlay_h, x:x+overlay_w] = blended.astype(np.uint8)

strt()