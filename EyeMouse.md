# Eye Mouse
The following script allows a user with a webcam to move their mouse, left-click and right-click by tilting their head and winking. This script was created because many solutions allowing those without the use of their hands to utilize a computer take the form of hardware that may be prohibitively expensive. So, a script such as this one could present a cost-effective alternative, especially for those who use computers casually and may not need the fine controls offered by special-built hardware.

```python
from imutils import face_utils
import dlib
import cv2
import pyautogui
import numpy as np
import math
import winsound

#This script enables a user to control their mouse by tilting their head and blinking their left and right eyes.
#Controls as follows:
#Tilt head to move the cursor, wink left-eye to right-click, wink right-eye to double-click. Hold right-eye closed to recenter head position.
#Press ESC to exit the program. (This may be done with only the mouse by using a virtual keyboard.)

class center:
    def __init__(self, centerShape, frameSize):
        self.centerX = centerShape[0]
        self.centerY = centerShape[1]
        self.centerCoord = (int(self.centerX/frameSize), int(self.centerY/frameSize))
        
def findEARatio(shape, right):
    if right==True:
        distOuterLid = math.sqrt((shape[37,0]-shape[41,0])**2 + (shape[37,1]-shape[41,1])**2)
        distInnerLid = math.sqrt((shape[38,0]-shape[40,0])**2 + (shape[38,1]-shape[40,1])**2)
        distAcrossEye = math.sqrt((shape[36,0]-shape[39,0])**2 + (shape[36,1]-shape[39,1])**2)
        eARatio=(distOuterLid+distInnerLid)/(distAcrossEye*2)
        return eARatio
    if right==False:
        distOuterLid = math.sqrt((shape[44,0]-shape[46,0])**2 + (shape[44,1]-shape[46,1])**2)
        distInnerLid = math.sqrt((shape[43,0]-shape[47,0])**2 + (shape[43,1]-shape[47,1])**2)
        distAcrossEye = math.sqrt((shape[42,0]-shape[45,0])**2 + (shape[42,1]-shape[45,1])**2)
        eARatio=(distOuterLid+distInnerLid)/(distAcrossEye*2)
        return eARatio

def drawMovementGraphics(frame, centerCoord, shape, frameSize):
    cv2.circle(frame, centerCoord, 2, (0,0,255), -1)
    cv2.line(frame, (int(shape[39,0]/frameSize),int(shape[39,1]/frameSize)), centerCoord, (255,0,0),2)

def reCenter(shape, frameSize):
    newCenter = center(shape, frameSize)
    return newCenter
    
video_capture = cv2.VideoCapture(0)

#Model path for facial landmark recognition
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


###PARAMETERS###
#Parameter for scaling the size of the processed image, in order to increase performance.
frameSize = 0.5
#Parameter for skipping processing on frames, in order to increase performance.
skipFrames = 1
#Paramter for skipping cursor movement on frames, in order to increase performance.
skipMoveCursor = 0
#Parameter for cursor movement speed.
mouseVel=0.4
#Paramter to boost movement along y-axis.
mouseYBalance=1.8
#Parameters for blink recognition ratio and duration necessary for a blink to be recognized(in frames as scalar of skipFrames)
blinkRatio = 0.22
unblinkRatio = 0.27
blinkDuration=1
#Parameter for how long to keep eye closed in order to re-center.
reCenterBlinkDuration=8

#List of facial landmarks to use
faceLandmarks=range(36,48)

###Setting indices and controls to default.###
skipIndex=0
rblinkIndex=0
lblinkIndex=0
moveCursorIndex=0
rblinkControl=False
lblinkControl=False
reCenterControl=False
centered=False

while True:
#If skipIndex has reached the skipFrames parameter, process and display image, then set the index to 0.
    if skipIndex==skipFrames:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0,0), fx=frameSize, fy=frameSize)
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray_frame, 0)
        #If a face is found, find facial indicators
        if rects:
            shape=predictor(gray_frame,rects[0])
            shape=face_utils.shape_to_np(shape)
            #From facial indicators, display those that we will use.
            for landmark in faceLandmarks:
                cv2.circle(frame, (int(shape[landmark,0]/frameSize),int(shape[landmark,1]/frameSize)), 2, (0,255,0), -1)


            ###MOVE CURSOR ACCORDING TO TEAR DUCT POSITION RELATIVE TO CENTER.###
            #If a center has been defined, draw it, and draw a line from the center to the tear duct.
            if centered == True:
                drawMovementGraphics(frame, centerCoord, shape, frameSize)

                #If the moveCursorIndex has reached the skipMoveCursor parameter, then move the cursor in accordance with vector between the tear duct and the center, and reset the index. If not, increment the index.
                if moveCursorIndex==skipMoveCursor:
                    eyeCenter = center(shape[39], frameSize)
                    difVectorX = eyeCenter.centerX-centerX
                    difVectorY = eyeCenter.centerY-centerY
                    pyautogui.moveRel(difVectorX*mouseVel, difVectorY*mouseVel*mouseYBalance, duration=0)
                    moveCursorIndex=0
                else:
                    moveCursorIndex+=1

            ###WHEN USER WINKS LEFT EYE, RIGHT-CLICK.###
            lEARatio = findEARatio (shape, False)
            #If the Eye Aspect Ratio is below parameter, begin counting towards recognizing purposeful blink.
            if lEARatio < blinkRatio:
                    #If blink has sufficient duration to be purposeful, double-click and activate blinkControl.
                    if lblinkIndex==blinkDuration and lblinkControl==False:
                        pyautogui.rightClick()
                        lblinkControl=True
                        lblinkIndex+=1
                        winsound.Beep(2000,500)
                    else:
                        lblinkIndex+=1
            #If the Eye Aspect Ratio is above the unblinkRatio parameter, reset the blinkControl to allow next blink to be recognized.
            #A separate parameter is used in order to reduce false negative blink detection for releasing the control without reducing sensitivity to blinking.
            elif lEARatio>unblinkRatio:
                lblinkIndex=0
                lblinkControl=False
                
            ###WHEN USER WINKS RIGHT EYE, DOUBLE-CLICK. WHEN HELD, RECENTER###        
            #Find Eye Aspect Ratio
            rEARatio = findEARatio(shape, True)

            #If the Eye Aspect Ratio is below parameter, begin counting towards recognizing purposeful blink.
            if lEARatio >= blinkRatio and rEARatio < blinkRatio and reCenterControl==False:
                    #If blink has sufficient duration to be purposeful, double-click and activate blinkControl.
                    if rblinkIndex==blinkDuration and rblinkControl==False:
                        pyautogui.doubleClick()
                        rblinkControl=True
                        rblinkIndex+=1
                        winsound.Beep(2500,500)
                    #If blink has sufficient duration to be a purposeful recenter, then do so.
                    if rblinkIndex==reCenterBlinkDuration:
                        newCenter = reCenter(shape[39], frameSize)
                        centerX = newCenter.centerX
                        centerY = newCenter.centerY
                        centerCoord = newCenter.centerCoord
                        centered = True
                        reCenterControl=True
                        winsound.Beep(2500,1500)
                    else:
                        rblinkIndex+=1
            #If the Eye Aspect Ratio is above the unblinkRatio parameter, reset the blinkControl to allow next blink to be recognized.
            #A separate parameter is used in order to reduce false negative blink detection for releasing the control without reducing sensitivity to blinking.
            elif rEARatio>unblinkRatio:
                rblinkIndex=0
                rblinkControl=False
                reCenterControl=False
                 
        cv2.imshow('Video',frame)
        skipIndex=0

#If skipIndex has not reached the skipFrames parameter, increment it by 1.
    else:
        skipIndex += 1
#When user presses Escape key, quit the program.      
    if cv2.waitKey(1) == 27:
        break
    
video_capture.release()
cv2.destroyAllWindows()
```
