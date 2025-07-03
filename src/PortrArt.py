################################################################################################################################
'''
PortrArt
Shreyan Deo
sdeo2
Carnegie Mellon University

Drawing Program

Credits: 
External Casacades - https://github.com/sightmachine
PIL Library Documentation - https://pillow.readthedocs.io/en/stable/
OS Library Documentation - https://docs.python.org/3/library/os.html
CircleCurvyLines.png - Made it myself
OvalCurvyLines.png - Made it myself
TriangleCurvyLines.png - Made it Myself
Color_Picker.png - ChatGPT
Flood Fill DFS Stacks Logic - https://www.youtube.com/watch?v=Sbciimd09h4&ab_channel=howCode
                            - Refining help from ChatGPT
Example1.jpeg - https://www.domestika.org/en/blog/6923-10-free-websites-for-portrait-reference-photos
Example2.jpeg - https://newravenna.fandom.com/wiki/Patricia_Toarina
Example3.jpeg - https://in.pinterest.com/pin/602637993883827696/ 
Example4.jpeg - https://www.freepik.com/premium-photo/happy-young-woman-with-natural-hair_20043392.htm
Example5.jpeg - https://www.andrew.cmu.edu/user/gkesden/oldscsstuff/ta/seminarabstracts/seriousfun.html
Example6.jpeg - https://www.cmu.edu/math/people/faculty/mackey.html
Example7.jpeg - https://www.poshenloh.com/
Example8.jpeg - https://in.pinterest.com/pin/422071796331422880/
Example9.jpeg - https://westcottu.com/capturing-subjects-with-glasses-using-the-eyelighter-2
Example10.jpeg - https://medium.com/the-happiness-of-pursuit/day-9-drawing-a-realistic-portrait-cdd6d972b761
userDrawing.png - drawing that the user makes
'''
################################################################################################################################

from cmu_graphics import *
from PIL import Image, ImageDraw
import math
import cv2
import os

################################################################################################################################
'''
Cascading Functions:
Face Detection
Eye Detection
Nose Detection
Mouth Detection
'''
################################################################################################################################

def faceDetection(app, gray):

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Find the largest face
    champArea = 0
    champX, champY, champL, champH = 0, 0, 0, 0
    for x, y, length, height in faces:
        area = length * height
        if area > champArea:  # Compare areas
            champArea = area
            champX, champY, champL, champH = x, y, length, height
    
    # Resize the coordinates based on the resize factor and relocate them
    faceCoordinates = (
        int(champX / app.resizeFactor),
        int(champY / app.resizeFactor),
        int(champL / app.resizeFactor),
        int(champH / app.resizeFactor),
    )
    return faceCoordinates

def eyeRegionDetection(app, gray, faceCoordinates):
    # Detect eye regions in the image
    eyeRegion_cascade = cv2.CascadeClassifier("../External_Cascades/external_two_eyes_big_cascade.xml")
    
    # Extract the face region based on the face coordinates
    faceX_resized, faceY_resized, faceL_resized, faceH_resized = resizeCoordinates(app, faceCoordinates)
    faceROI = gray[faceY_resized:faceY_resized + faceH_resized, faceX_resized:faceX_resized + faceL_resized]

    # Detect eyes within the face region
    eyeRegion = eyeRegion_cascade.detectMultiScale(faceROI, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    # Resize the eye region coordinates based on the resize factor
    eyeRegionCoordinates = []
    for (ex, ey, ew, eh) in eyeRegion:
        eyeRegionCoordinates.append((
            int((faceX_resized + ex) / app.resizeFactor),
            int((faceY_resized + ey) / app.resizeFactor),
            int(ew / app.resizeFactor),
            int(eh / app.resizeFactor)
        ))

    # Find the largest eye region
    champArea = 0
    champX, champY, champL, champH = 0, 0, 0, 0
    for (x, y, length, height) in eyeRegionCoordinates:
        area = length * height
        if area > champArea:  # Compare areas
            champArea = area
            champX, champY, champL, champH = x, y, length, height
    eyeRegion = (champX, champY, champL, champH)

    return eyeRegion

def eyeDetection(app, gray, faceCoordinates):

    # Load the Haar cascade for eye detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Extract the face region based on the face coordinates
    faceX_resized, faceY_resized, faceL_resized, faceH_resized = resizeCoordinates(app, faceCoordinates)
    faceROI = gray[faceY_resized:faceY_resized + faceH_resized, faceX_resized:faceX_resized + faceL_resized]

    # Detect noses within the face region
    eyes = eye_cascade.detectMultiScale(faceROI)

    # Resize the nose coordinates based on the resize factor
    eyeCoordinates = []
    for (ex, ey, ew, eh) in eyes:
        eyeCoordinates.append((
            int((faceX_resized + ex) / app.resizeFactor),
            int((faceY_resized + ey) / app.resizeFactor),
            int(ew / app.resizeFactor),
            int(eh / app.resizeFactor)
        ))

    return eyeCoordinates

def noseDetection(app, gray, faceCoordinates):

    # Load the Haar cascade for nose detection
    nose_cascade = cv2.CascadeClassifier("../External_Cascades/external_nose_cascade.xml")

    # Extract the face region based on the face coordinates
    faceX_resized, faceY_resized, faceL_resized, faceH_resized = resizeCoordinates(app, faceCoordinates)
    faceROI = gray[faceY_resized:faceY_resized + faceH_resized, faceX_resized:faceX_resized + faceL_resized]

    # Detect noses within the face region
    noses = nose_cascade.detectMultiScale(faceROI, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    # Resize the nose coordinates based on the resize factor
    noseCoordinates = []
    for (nx, ny, nw, nh) in noses:
        noseCoordinates.append((
            int((faceX_resized + nx) / app.resizeFactor),
            int((faceY_resized + ny) / app.resizeFactor),
            int(nw / app.resizeFactor),
            int(nh / app.resizeFactor)
        ))

    return noseCoordinates

def mouthDetection(app, gray, faceCoordinates):

    # Load the Haar cascade for mouth detection
    mouth_cascade = cv2.CascadeClassifier("../External_Cascades/external_mouth_cascade.xml")

    # Extract the face region based on the face coordinates
    faceX_resized, faceY_resized, faceL_resized, faceH_resized = resizeCoordinates(app, faceCoordinates)
    faceROI = gray[faceY_resized:faceY_resized + faceH_resized, faceX_resized:faceX_resized + faceL_resized]

    # Detect mouths within the face region
    mouths = mouth_cascade.detectMultiScale(faceROI, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

    # Resize the mouth coordinates based on the resize factor
    mouthCoordinates = []
    for (mx, my, mw, mh) in mouths:
        # Adjust the position relative to the original image
        mouthCoordinates.append((
            int((faceX_resized + mx) / app.resizeFactor),
            int((faceY_resized + my) / app.resizeFactor),
            int(mw / app.resizeFactor),
            int(mh / app.resizeFactor)
        ))

    return mouthCoordinates

################################################################################################################################
'''
Feature Filters:
Eye Filter
Nose Filter
Mouth Filter
'''
################################################################################################################################

def eyesFilter(app, realEyeCoordinates):
    # Find two biggest rectangles
    # Initialize variables for the largest and second largest eye regions
    largestArea = 0
    largestX, largestY, largestL, largestH = 0, 0, 0, 0

    secondLargestArea = 0
    secondLargestX, secondLargestY, secondLargestL, secondLargestH = 0, 0, 0, 0

    # Iterate through the eye region coordinates
    for (x, y, length, height) in realEyeCoordinates:
        area = length * height
        if area > largestArea:  # New largest region found
            # Update second largest before replacing the largest
            secondLargestArea = largestArea
            secondLargestX, secondLargestY, secondLargestL, secondLargestH = (largestX, largestY, largestL, largestH)
            # Update the largest region
            largestArea = area
            largestX, largestY, largestL, largestH = x, y, length, height
        elif area > secondLargestArea:  # New second largest region found
            secondLargestArea = area
            secondLargestX, secondLargestY, secondLargestL, secondLargestH = (x, y, length, height)

    # Store the results in tuples
    largestEyeRegion = (largestX, largestY, largestL, largestH)
    secondLargestEyeRegion = (secondLargestX, secondLargestY, secondLargestL, secondLargestH)

    return [largestEyeRegion, secondLargestEyeRegion]
    
def noseFilter(app, filteredEyesBeforeCorrection, noseCoordinates):
    # Remove all rectangles above eyes
    # Find biggest rectangle
    noseFilterList=[]
    for x,y,w,h in filteredEyesBeforeCorrection:
        for x1,y1,w1,h1 in noseCoordinates:
            if y+h/2 < y1:
                noseFilterList.append((x1,y1,w1,h1))

    champArea = 0
    champX, champY, champL, champH = 0, 0, 0, 0
    for x, y, w, h in noseFilterList:
        area = w * h
        if area > champArea:  # Compare areas
            champArea = area
            champX, champY, champL, champH = x, y, w, h
    return [(champX, champY, champL, champH)]

def mouthFilter(app, filteredNoseBeforeCorrection, mouthCoordinates):
    # Remove all rectangles above nose
    # Find biggest rectangle
    mouthFilterList=[]
    for x,y,w,h in filteredNoseBeforeCorrection:
        for x1,y1,w1,h1 in mouthCoordinates:
            if y+h/2 < y1:
                mouthFilterList.append((x1,y1,w1,h1))

    champArea = 0
    champX, champY, champL, champH = 0, 0, 0, 0
    for x, y, w, h in mouthFilterList:
        area = w * h
        if area > champArea:  # Compare areas
            champArea = area
            champX, champY, champL, champH = x, y, w, h
    return [(champX, champY, champL, champH)]

################################################################################################################################
'''
OnAppStart
'''
################################################################################################################################

def onAppStart(app):

    ############################################################################################################################
    '''
    Drawing Program
    '''
    ############################################################################################################################


    screenWidth, screenHeight = 1600*0.8, 900*0.8
    app.width = int(screenWidth)
    app.height = int(screenHeight)

    app.completeBrushStrokesList = []
    app.currentBrushStrokesList = []
    app.lastBrushStroke = []

    # Default Tool Settings
    app.currentTool = 'brush'
    app.brushSize = 5
    app.brushShape = 'oval'
    app.brushColor = (0, 0, 0)

    app.circleCurvyLines = "../UI_Images/CircleCurvyLines.png"
    app.ovalCurvyLines = "../UI_Images/OvalCurvyLines.png"
    app.triangleCurvyLines = "../UI_Images/TriangleCurvyLines.png"
    app.colorPicker = "../UI_Images/Color_Picker.png"

    updateSizes(app)
    app.canvasColor = (255, 255, 255)
    app.pilCanvas = createCanvas(app.canvasWidth, app.canvasHeight, app.canvasColor)
    app.pilColorPicker = createCanvas(int(app.colorWheelWidth), int(app.colorWheelHeight), app.canvasColor)
    image = Image.open(app.colorPicker)
    image = image.resize((int(app.colorWheelWidth), int(app.colorWheelHeight)))
    app.pilColorPicker.paste(image, (0, 0))

    numOfButtonsPerRow = 3
    numOfRows = 1
    app.brushSizeIcons = [('circle',7), ('circle',5), ('circle',3)]
    app.brushSizeButtons = getButtons(numOfButtonsPerRow, numOfRows, app.brushSettingsX, app.brushSettingsY, app.brushSettingsWidth, app.brushSettingsHeight/3)

    numOfButtonsPerRow = 3
    numOfRows = 1
    app.brushShapeIcons = [('circle', 7), ('oval', 7), ('triangle', 7)]
    app.brushShapeButtons = getButtons(numOfButtonsPerRow, numOfRows, app.brushSettingsX, app.brushSettingsY + 2*app.brushSettingsHeight/3, app.brushSettingsWidth, app.brushSettingsHeight/3)

    numOfButtonsPerRow = 6
    numOfRows = 1
    app.toolBarIcons = ['Brush','Eraser','Flood Fill','Dropper','Upload','Save']
    app.toolBarButtons = getButtons(numOfButtonsPerRow, numOfRows, app.toolBarX, app.toolBarY, app.toolBarWidth, app.toolBarHeight)

    numOfButtonsPerRow = 1
    numOfRows = 1
    app.portraitGuideIcons = ['Portrait Guide']
    app.portraitGuideButtons = getButtons(numOfButtonsPerRow, numOfRows, app.toolsX, app.toolsY, app.toolsWidth, app.toolsHeight/6)

    numOfButtonsPerRow = 2
    numOfRows = 2
    app.portraitGuideToolIcons = ['Image', 'Boxes', 'Lines', 'Dots']
    app.portraitGuideToolButtons = getButtons(numOfButtonsPerRow, numOfRows, app.toolsX, app.toolsY + app.toolsHeight/6 - app.toolsHeight/18, app.toolsWidth, app.toolsHeight/3)

    numOfButtonsPerRow = 1
    numOfRows = 1
    app.accuracyCheckerIcons = ['Test Accuracy']
    app.accuracyCheckerButtons = getButtons(numOfButtonsPerRow, numOfRows, app.toolsX, app.toolsY + app.toolsHeight/6 - app.toolsHeight/18 + app.toolsHeight/3, app.toolsWidth, app.toolsHeight/6)

    app.imageGuide = True
    app.boxesGuide = True
    app.linesGuide = True
    app.dotsGuide = True

    app.labelList = []


    ############################################################################################################################
    '''
    Face Detection
    '''
    ############################################################################################################################

    app.imgPath = "../15112_Face_Detection_Images/Example5.jpeg"
    doAllFaceCalculations(app)

    app.canvasCentre = app.canvasX + app.canvasWidth/2
    app.OffsetXToDrawingCanvas = app.userImageSpaceCentreX - app.canvasCentre

    app.uploadWindowCentreX, app.uploadWindowCentreY, app.uploadWindowWidth, app.uploadWindowHeight = app.width/2, app.height/2, app.width/2, app.height/2
    app.uploadWindowX, app.uploadWindowY = app.uploadWindowCentreX - app.uploadWindowWidth/2, app.uploadWindowCentreY - app.uploadWindowHeight/2

    numOfButtonsPerRow = 2
    numOfRows = 2
    app.uploadIcons = ['Enter Reference Image:', '', 'Enter Base Image:', '']
    app.uploadButtons = getButtons(numOfButtonsPerRow, numOfRows, app.uploadWindowX, app.uploadWindowY, app.uploadWindowWidth, 2*app.uploadWindowHeight/3)

    numOfButtonsPerRow = 1
    numOfRows = 1
    app.uploadDoneIcons = ['Done']
    app.uploadDoneButtons = getButtons(numOfButtonsPerRow, numOfRows, app.uploadWindowX, app.uploadWindowY + 2*app.uploadWindowHeight/3, app.uploadWindowWidth, app.uploadWindowHeight/3)

    app.referenceImageName = ''
    app.baseImageName = ''
    app.writingReferenceImageName = False
    app.writingBaseImageName = False



################################################################################################################################
'''
Mouse Events:
onMousePress
onMouseDrag
'''
################################################################################################################################

def onMousePress(app, mouseX, mouseY):
    app.currentBrushStrokesList = []

    # Color Selection
    if app.currentTool == 'brush' or app.currentTool == 'floodFill':
        if app.colorWheelX<=mouseX<=app.colorWheelX+app.colorWheelWidth and app.colorWheelY<=mouseY<=app.colorWheelY+app.colorWheelHeight:
            color = getColorAt(app.pilColorPicker, mouseX, mouseY)
            app.brushColor = color

    # Tool Selection
    toolChoice = ['brush','eraser','floodFill','dropper','upload','save']
    for i in range(len(app.toolBarButtons)):
        brushButton = app.toolBarButtons[i]
        x, y, width, height = brushButton
        if x<=mouseX<=x+width and y<=mouseY<=y+height:
                app.currentTool = toolChoice[i]
    
    # Guide Tool Selection
    for i in range(len(app.portraitGuideToolButtons)):
        brushButton = app.portraitGuideToolButtons[i]
        x, y, width, height = brushButton
        if x<=mouseX<=x+width and y<=mouseY<=y+height:
                if i == 0:
                    app.imageGuide = not(app.imageGuide)
                elif i == 1:
                    app.boxesGuide = not(app.boxesGuide)
                elif i == 2:
                    app.linesGuide = not(app.linesGuide)
                elif i == 3:
                    app.dotsGuide = not(app.dotsGuide)


    # Brush/Eraser Size Buttons
    if app.currentTool == 'brush' or app.currentTool == 'eraser':
        size7Button = app.brushSizeButtons[0]
        x, y, width, height = size7Button
        if x<=mouseX<=x+width and y<=mouseY<=y+height:
            app.brushSize = 7

        size5Button = app.brushSizeButtons[1]
        x, y, width, height = size5Button
        if x<=mouseX<=x+width and y<=mouseY<=y+height:
            app.brushSize = 5

        size3Button = app.brushSizeButtons[2]
        x, y, width, height = size3Button
        if x<=mouseX<=x+width and y<=mouseY<=y+height:
            app.brushSize = 3

    # Brush/Eraser Shape Buttons
    if app.currentTool == 'brush' or app.currentTool == 'eraser':
        circleButton = app.brushShapeButtons[0]
        x, y, width, height = circleButton
        if x<=mouseX<=x+width and y<=mouseY<=y+height:
            app.brushShape = 'circle'

        ovalButton = app.brushShapeButtons[1]
        x, y, width, height = ovalButton
        if x<=mouseX<=x+width and y<=mouseY<=y+height:
            app.brushShape = 'oval'

        triangleButton = app.brushShapeButtons[2]
        x, y, width, height = triangleButton
        if x<=mouseX<=x+width and y<=mouseY<=y+height:
            app.brushShape = 'triangle'
    
    # Color Dropper
    if app.currentTool == 'dropper':
        if app.canvasX <= mouseX <= app.canvasX + app.canvasWidth and app.canvasY <= mouseY <= app.canvasY+app.canvasHeight:
            canvasX = mouseX - app.canvasX
            canvasY = mouseY - app.canvasY
            app.brushColor = getColorAt(app.pilCanvas, canvasX, canvasY)
            app.currentTool = 'brush'
        elif app.imgX <= mouseX <= app.imgX + app.imgWidth and app.imgY <= mouseY <= app.imgY + app.imgHeight:
            imageCanvasX = mouseX - app.imgX
            imageCanvasY = mouseY - app.imgY
            app.brushColor = getColorAt(app.pilUserImageCanvas, imageCanvasX, imageCanvasY)
            app.currentTool = 'brush'

    # Flood Fill
    if app.currentTool == 'floodFill':
        canvasX = mouseX - app.canvasX
        canvasY = mouseY - app.canvasY

        if 0 <= canvasX < app.canvasWidth and 0 <= canvasY < app.canvasHeight:
            targetColor = app.pilCanvas.getpixel((canvasX, canvasY))
            floodFill(app, canvasX, canvasY, targetColor, app.brushColor)

    # Upload Tool Functionality
    if app.currentTool == 'upload':
        for i in range(len(app.uploadButtons)):
            brushButton = app.uploadButtons[i]
            x, y, width, height = brushButton
            if x<=mouseX<=x+width and y<=mouseY<=y+height:
                    if i == 1:
                        app.writingReferenceImage = True
                        app.writingBaseImage = False
                        break
                    elif i == 3:
                        app.writingBaseImage = True 
                        app.writingReferenceImage = False
                        break
        else:
            app.writingReferenceImage = False
            app.writingBaseImage = False
            x, y, width, height = app.uploadDoneButtons[0]
            if x <= mouseX <= x+width and y <= mouseY <= y+height:
                imgPath1 = '../15112_Face_Detection_Images/'+app.referenceImageName+'.jpeg'
                imgPath2 = '../15112_Face_Detection_Images/'+app.baseImageName+'.jpeg'
                if os.path.exists(imgPath1):
                    app.imgPath = imgPath1
                    doAllFaceCalculations(app)
                if os.path.exists(imgPath2):
                    imgWidth2, imgHeight2, resizeFactor = imageResizer(imgPath2, app.canvasWidth, app.canvasHeight)
                    image = Image.open(imgPath2)
                    image = image.resize((int(imgWidth2), int(imgHeight2)))
                    app.pilCanvas.paste(image, ((app.canvasWidth - imgWidth2)//2, (app.canvasHeight - imgHeight2)//2))
                app.referenceImageName = ''
                app.baseImageName = ''
                app.currentTool = 'brush'
    
    # Save Functionality
    if app.currentTool == 'save':
        directory = "../15112_Face_Detection_Images"
        filename = "userDrawing.png"
        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the drawing
        filepath = os.path.join(directory, filename)
        app.pilCanvas.save(filepath)
        app.currentTool = 'brush'
        
    # Accuracy Checker
    accuracyCheckerButtonsX, accuracyCheckerButtonsY, accuracyCheckerButtonsWidth, accuracyCheckerButtonsHeight = app.accuracyCheckerButtons[0]
    if accuracyCheckerButtonsX <= mouseX <= accuracyCheckerButtonsX + accuracyCheckerButtonsWidth and accuracyCheckerButtonsY <= mouseY <= accuracyCheckerButtonsY + accuracyCheckerButtonsHeight:
        directory = "../15112_Face_Detection_Images"
        filename = "temp.png"
        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the drawing
        filepath = os.path.join(directory, filename)
        app.pilCanvas.save(filepath)
        app.imgPath2 = "../15112_Face_Detection_Images/temp.png"
        doAllFaceCalculationsForDrawingCanvas(app)
        app.labelList = []
        if app.faceCoordinates2 == (0, 0, 0, 0) or app.filteredLeftEye == [(0, 0, 0, 0)] or app.filteredRightEye == [(0, 0, 0, 0)] or app.filteredNose == [(0, 0, 0, 0)] or app.filteredMouth == [(0, 0, 0, 0)] or\
            app.faceCoordinates2[3] == 0 or app.filteredLeftEye2[0][3] == 0 or app.filteredRightEye2[0][3] == 0 or app.filteredNose2[0][3] == 0 or app.filteredMouth2[0][3] == 0:
            app.labelList.append('Face Not Found in Drawing')
        else:
            originalRatio = app.faceCoordinates[2]/app.faceCoordinates[3]
            newRatio = app.faceCoordinates2[2]/app.faceCoordinates2[3]
            faceAccuracy = (1-((abs(originalRatio-newRatio))/originalRatio))*100
            app.labelList.append('Face Accuracy: '+str(faceAccuracy))

            originalRatio = app.filteredLeftEye[0][2]/app.filteredLeftEye[0][3]
            newRatio = app.filteredLeftEye2[0][2]/app.filteredLeftEye2[0][3]
            leftEyeAccuracy = (1-((abs(originalRatio-newRatio))/originalRatio))*100
            app.labelList.append('Left Eye Accuracy: '+str(leftEyeAccuracy))

            originalRatio = app.filteredRightEye[0][2]/app.filteredRightEye[0][3]
            newRatio = app.filteredRightEye2[0][2]/app.filteredRightEye2[0][3]
            rightEyeAccuracy = (1-((abs(originalRatio-newRatio))/originalRatio))*100
            app.labelList.append('Right Eye Accuracy: '+str(rightEyeAccuracy))

            originalRatio = app.filteredNose[0][2]/app.filteredNose[0][3]
            newRatio = app.filteredNose2[0][2]/app.filteredNose2[0][3]
            noseAccuracy = (1-((abs(originalRatio-newRatio))/originalRatio))*100
            app.labelList.append('Nose Accuracy: '+str(noseAccuracy))

            originalRatio = app.filteredMouth[0][2]/app.filteredMouth[0][3]
            newRatio = app.filteredMouth2[0][2]/app.filteredMouth2[0][3]
            mouthAccuracy = (1-((abs(originalRatio-newRatio))/originalRatio))*100
            app.labelList.append('Mouth Accuracy: '+str(mouthAccuracy))
            
        os.remove(filepath)
        
    
    



def onMouseDrag(app, mouseX, mouseY):
    canvasX = mouseX - app.canvasX
    canvasY = mouseY - app.canvasY
    
    if app.currentTool == 'brush' or app.currentTool == 'eraser':
        if app.currentTool == 'brush':
            brushColor = app.brushColor
        elif app.currentTool == 'eraser':
            brushColor = app.canvasColor
        if 0 <= canvasX < app.canvasWidth and 0 <= canvasY < app.canvasHeight:
            draw = ImageDraw.Draw(app.pilCanvas)
            if app.currentBrushStrokesList:
                lastX, lastY, _, _ = app.currentBrushStrokesList[-1]
                points = interpolatePoints(lastX, lastY, canvasX, canvasY)
                for x, y in points:
                    if app.brushShape == 'circle':
                        drawPILCircle(draw, x, y, app.brushSize, brushColor) # PIL Circle
                    elif app.brushShape == 'oval':
                        drawPILOval(draw, x, y, app.brushSize, brushColor) # PIL Oval
                    elif app.brushShape == 'triangle':
                        drawPILTriangle(draw, x, y, app.brushSize, brushColor) # PIL Triangle
                    app.currentBrushStrokesList.append((x, y, app.brushSize, brushColor))
                    app.completeBrushStrokesList += app.currentBrushStrokesList
            else:
                if app.brushShape == 'circle':
                    drawPILCircle(draw, canvasX, canvasY, app.brushSize, brushColor) # PIL Circle
                elif app.brushShape == 'oval':
                    drawPILOval(draw, canvasX, canvasY, app.brushSize, brushColor) # PIL Oval
                elif app.brushShape == 'triangle':
                    drawPILTriangle(draw, canvasX, canvasY, app.brushSize, brushColor) # PIL Triangle
                app.currentBrushStrokesList.append((canvasX, canvasY, app.brushSize, brushColor))
                app.completeBrushStrokesList += app.currentBrushStrokesList


################################################################################################################################
'''
On Key Press
'''
################################################################################################################################

def onKeyPress(app, key):
    if app.writingReferenceImage:
        if key == 'backspace':
            app.referenceImageName = app.referenceImageName[:-1]
        else:
            app.referenceImageName += key
    if app.writingBaseImage:
        if key == 'backspace':
            app.baseImageName = app.referenceImageName[:-1]
        else:
            app.baseImageName += key



################################################################################################################################
'''
Redraw All
'''
################################################################################################################################

def redrawAll(app):

    ############################################################################################################################
    '''
    Drawing Program
    '''
    ############################################################################################################################

    # Color Wheel
    drawRect(app.colorWheelX, app.colorWheelY, app.colorWheelWidth, app.colorWheelHeight, fill = None, border = 'red')

    # Draw Color Picker
    cmuColorPicker = CMUImage(app.pilColorPicker)
    drawImage(cmuColorPicker, app.colorWheelX, app.colorWheelY)

    # Brush Size
    drawRect(app.brushSettingsX, app.brushSettingsY, app.brushSettingsWidth, app.brushSettingsHeight, fill = None, border = 'blue')

    # Brush Size Buttons
    drawButtons(app.brushSizeButtons, app.brushSizeIcons)

    # Curvy Lines
    if app.brushShape == 'circle':
        imageWidth, imageHeight = getImageSize(app.circleCurvyLines)
        drawImage(app.circleCurvyLines, app.brushSettingsX + app.brushSettingsWidth/9, app.brushSettingsY + app.brushSettingsHeight/3 - 15, width=19*imageWidth//24, height=19*imageHeight//24)
    
    elif app.brushShape == 'oval':
        imageWidth, imageHeight = getImageSize(app.ovalCurvyLines)
        drawImage(app.ovalCurvyLines, app.brushSettingsX + app.brushSettingsWidth/9, app.brushSettingsY + app.brushSettingsHeight/3 - 15, width=19*imageWidth//24, height=19*imageHeight//24)

    elif app.brushShape == 'triangle':
        imageWidth, imageHeight = getImageSize(app.triangleCurvyLines)
        drawImage(app.triangleCurvyLines, app.brushSettingsX + app.brushSettingsWidth/9, app.brushSettingsY + app.brushSettingsHeight/3 - 15, width=19*imageWidth//24, height=19*imageHeight//24)

    # Brush Shape Buttons
    drawButtons(app.brushShapeButtons, app.brushShapeIcons)

    # Tools
    drawRect(app.toolsX, app.toolsY, app.toolsWidth, app.toolsHeight, fill = None, border = 'yellow')

    # Tool Buttons
    drawButtons(app.portraitGuideButtons, app.portraitGuideIcons)
    drawButtons(app.portraitGuideToolButtons, app.portraitGuideToolIcons)
    drawButtons(app.accuracyCheckerButtons, app.accuracyCheckerIcons)
    if app.labelList:
        for labelIndex in range(len(app.labelList)):
            label = app.labelList[labelIndex]
            drawLabel(label, app.toolsX+app.toolsWidth/2, app.toolsY + ((labelIndex+1)/6)*(1/3)*app.toolsHeight + 2*app.toolsHeight/6 - app.toolsHeight/18 + app.toolsHeight/3)
        

    # Tool Bar
    drawRect(app.toolBarX, app.toolBarY, app.toolBarWidth, app.toolBarHeight, fill = None, border = 'pink')

    # Tool Bar Buttons
    drawButtons(app.toolBarButtons, app.toolBarIcons)

    # Canvas
    cmuCanvas = CMUImage(app.pilCanvas)
    drawImage(cmuCanvas, app.canvasX, app.canvasY)

    # User Image Space
    drawRect(app.userImageSpaceX, app.userImageSpaceY, app.userImageSpaceWidth, app.userImageSpaceHeight, fill = None, border = 'lightBlue')


    ############################################################################################################################
    '''
    Face Detection
    '''
    ############################################################################################################################

    # Draw the image
    cmuUserImage = CMUImage(app.pilUserImageCanvas)
    drawImage(cmuUserImage, app.imgX, app.imgY)
    
    drawFacialFeatureHelpers(app)

    # Buffer Windows

    # Upload Window
    if app.currentTool == 'upload':
        drawRect(app.uploadWindowX, app.uploadWindowY, app.uploadWindowWidth, app.uploadWindowHeight, fill = 'white', border = 'black')
        drawButtons(app.uploadButtons, app.uploadIcons)
        drawButtons(app.uploadDoneButtons, app.uploadDoneIcons)
        referenceImageTextX, referenceImageTextY, referenceImageTextWidth, referenceImageTextHeight = app.uploadButtons[1]
        drawLabel(app.referenceImageName, referenceImageTextX + referenceImageTextWidth/2, referenceImageTextY + referenceImageTextHeight/2, size = 12)
        baseImageTextX, baseImageTextY, baseImageTextWidth, baseImageTextHeight = app.uploadButtons[3]
        drawLabel(app.baseImageName, baseImageTextX + baseImageTextWidth/2, baseImageTextY + baseImageTextHeight/2, size = 12)

################################################################################################################################
'''
Drawing Program Helpers:
createCanvas - creates PIL Canvas
drawPILCircle - draws circle on PIL Canvas
drawPILOval - draws oval on PIL Canvas 
drawPILTriangle - draws triangle on PIL Canvas
updateSizes -  updates/creates the sizes of all the different features such as color picker, tool bar, etc.
interpolatePoints - adds points between the shapes when user is drawing to make the drawing smoother; currently set to 10 points
getButtons - return the places where the buttons should be placed in a region depending on how many buttons the user wants to place per row, and number of rows
drawButtons - draws the buttons by taking the coordinates, also draws the icons on the buttons
getTriangleCoordinates - returns the three vertices of the triangle that is required
getColorAt - get the color of the pixel at the coordinate entered
'''
################################################################################################################################

def createCanvas(canvasWidth, canvasHeight, bgColor):
    return Image.new('RGB', (canvasWidth, canvasHeight), bgColor)

def drawPILCircle(draw, x, y, radius, color):
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

def drawPILOval(draw, x, y, radius, color):
    draw.ellipse([x - radius/2, y - radius, x + radius/2, y + radius], fill=color)

def drawPILTriangle(draw, x, y, radius, color):
    triangleCoords = getTriangleCoordinates(x, y, radius)
    vertex1 = (triangleCoords[0], triangleCoords[1])
    vertex2 = (triangleCoords[2], triangleCoords[3])
    vertex3 = (triangleCoords[4], triangleCoords[5])
    triangle = [vertex1, vertex2, vertex3]
    draw.polygon(triangle, fill=color)

def updateSizes(app):
    # Color Wheel
    app.colorWheelX, app.colorWheelY = 0, 0
    app.colorWheelWidth, app.colorWheelHeight = app.height*0.3, app.height*0.3

    # Brush Settings
    app.brushSettingsX, app.brushSettingsY = 0, app.colorWheelY + app.colorWheelHeight
    app.brushSettingsWidth, app.brushSettingsHeight = app.height*0.3, app.height*0.3

    # Tools
    app.toolsX, app.toolsY = 0, app.brushSettingsY + app.brushSettingsHeight
    app.toolsWidth, app.toolsHeight = app.height*0.3, app.height*0.4

    # Tool Bar
    app.toolBarX, app.toolBarY = app.colorWheelX + app.colorWheelWidth, 0
    app.toolBarWidth, app.toolBarHeight = app.width-app.colorWheelWidth, app.height * 0.05

    # Canvas
    app.canvasX, app.canvasY = (app.colorWheelX + app.colorWheelWidth), (app.toolBarY + app.toolBarHeight)
    app.canvasWidth, app.canvasHeight = int((app.width-app.canvasX)/2), int(app.height-app.canvasY)

    # User Image
    app.userImageSpaceX, app.userImageSpaceY = app.canvasX + app.canvasWidth, app.canvasY 
    app.userImageSpaceWidth, app.userImageSpaceHeight = int((app.width-app.userImageSpaceX)), int((app.height-app.userImageSpaceY))

def interpolatePoints(x1, y1, x2, y2):
    points = []
    numPoints = 10
    for i in range(numPoints + 1):
        t = i / numPoints
        x = x1 * (1 - t) + x2 * t
        y = y1 * (1 - t) + y2 * t
        points.append((x, y))
    return points

def getButtons(numOfButtonsPerRow, numOfButtonRows, boxX, boxY, boxWidth, boxHeight):
    buttons = []
    widthBetweenButtons = boxWidth/(numOfButtonsPerRow*2+(numOfButtonsPerRow+1))
    heightBetweenButtons = boxHeight/(numOfButtonRows*2+(numOfButtonRows+1))
    buttonWidth = widthBetweenButtons*2
    buttonHeight = heightBetweenButtons*2

    for buttonRow in range(numOfButtonRows):
        for buttonNum in range(numOfButtonsPerRow):
            buttonX = widthBetweenButtons*(buttonNum+1) + buttonWidth*buttonNum + boxX
            buttonY = heightBetweenButtons*(buttonRow+1) + buttonHeight*buttonRow + boxY
            buttons.append((buttonX, buttonY, buttonWidth, buttonHeight))
    return buttons

def drawButtons(buttons, icons):
    iconIndex = 0
    for buttonX, buttonY, buttonWidth, buttonHeight in buttons:
        icon = icons[iconIndex]
        drawRect(buttonX, buttonY, buttonWidth, buttonHeight, fill = None, border = 'black')
        # Draw Icon
        # Text
        if type(icon)==str:
            x = buttonX + buttonWidth/2
            y = buttonY + buttonHeight/2
            drawLabel(icon, x, y, size = 15)

        # Shapes
        elif icon[0] == 'circle':
            circleX = buttonX + buttonWidth/2
            circleY = buttonY + buttonHeight/2
            circleR = icon[1]
            drawCircle(circleX, circleY, circleR)
    
        elif icon[0] == 'oval':
            ovalX = buttonX + buttonWidth/2
            ovalY = buttonY + buttonHeight/2
            ovalRX = icon[1]
            ovalRY = icon[1]*2
            drawOval(ovalX, ovalY, ovalRX, ovalRY)

        elif icon[0] == 'triangle':
            triangleCentreX = buttonX + buttonWidth/2
            triangleCentreY = buttonY + buttonHeight/2
            triangleRadius = icon[1]
            triangleCoordinates = getTriangleCoordinates(triangleCentreX, triangleCentreY, triangleRadius)
            drawPolygon(*triangleCoordinates)
        iconIndex += 1

def getTriangleCoordinates(x, y, a):
    sin_30 = math.sin(math.radians(30))
    cos_30 = math.cos(math.radians(30))
    sideLength = 2 * a * cos_30
    centreToBaseLength = a * sin_30
    topCoordinate = (x, y-a)
    bottomRightCoordinate = (x+sideLength/2, y+centreToBaseLength)
    bottomLeftCoordinate = (x-sideLength/2, y+centreToBaseLength)
    return topCoordinate+bottomRightCoordinate+bottomLeftCoordinate

def getColorAt(canvas, x, y):
    return canvas.getpixel((x, y))

def floodFill(app, startX, startY, targetColor, fillColor):

    if targetColor == fillColor:
        return
    
    stack = [(startX, startY)]
    visited = set()

    while stack:
        x, y = stack.pop()

        if (0 <= x < app.canvasWidth and 0 <= y < app.canvasHeight and (x, y) not in visited):
            currentColor = app.pilCanvas.getpixel((x, y))
            if currentColor == targetColor:

                draw = ImageDraw.Draw(app.pilCanvas)
                draw.point((x, y), fillColor)
                visited.add((x, y))

                stack.append((x+1, y))
                stack.append((x-1, y))
                stack.append((x, y+1))  
                stack.append((x, y-1))

################################################################################################################################
'''
Face Detection Helpers:
Image Resizer
Gray Scaler
Resize Coordinates
Check Intersecting Rectangles
Return Interecting Rectangles
'''
################################################################################################################################

def imageResizer(img_path, resizeMaxWidth, resizeMaxHeight):
    imgWidth, imgHeight = getImageSize(img_path)
    resizeFactor = 1

    if imgWidth == resizeMaxWidth and imgHeight == resizeMaxHeight:
        return imgWidth, imgHeight, resizeFactor
    
    elif imgWidth >= resizeMaxWidth or imgHeight >= resizeMaxHeight:
        while True:
            if imgWidth <= resizeMaxWidth and imgHeight <= resizeMaxHeight:
                return int(imgWidth), int(imgHeight), resizeFactor
            imgHeight /= 1.1
            imgWidth /= 1.1
            resizeFactor *= 1.1
    
    elif imgWidth < resizeMaxWidth and imgHeight < resizeMaxHeight:
        while True:
            if imgWidth > resizeMaxWidth or imgHeight > resizeMaxHeight:
                return int(imgWidth/1.1), int(imgHeight/1.1), resizeFactor*1.1
            imgHeight *= 1.1
            imgWidth *= 1.1
            resizeFactor /= 1.1

def grayScaleImage(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def resizeCoordinates(app, coordinates):
    resized_Coordinates = [int(app.resizeFactor * c) for c in coordinates]
    return resized_Coordinates

def intersect(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Calculate the intersection boundaries
    x_inter = max(x1, x2)
    y_inter = max(y1, y2)
    x_inter_end = min(x1 + w1, x2 + w2)
    y_inter_end = min(y1 + h1, y2 + h2)

    # Check if they intersect
    if x_inter < x_inter_end and y_inter < y_inter_end:
        return (x_inter, y_inter, x_inter_end - x_inter, y_inter_end - y_inter)
    return None

def rectIntersections(rectangles1, rectangles2):
    intersection = []
    for rect1 in rectangles1:
        for rect2 in rectangles2:
            intersection.append(intersect(rect1, rect2))
    realIntersection=[]
    for elem in intersection:
        if elem != None:
            realIntersection.append(elem)
    return realIntersection

def offsetImageDetectionParts(app, featureList):
    correctFacialFeatureList = []
    for facialFeatureTuple in featureList:
        facialFeatureX = facialFeatureTuple[0] + app.userImageSpaceX + app.imgOffsetX
        facialFeatureY = facialFeatureTuple[1] + app.userImageSpaceY + app.imgOffsetY
        correctFacialFeatureTuple = (facialFeatureX, facialFeatureY, facialFeatureTuple[2], facialFeatureTuple[3])
        correctFacialFeatureList.append(correctFacialFeatureTuple)
    return correctFacialFeatureList

def offsetImageFaceCoordinates(app, faceTuple):
    facialFeatureX = faceTuple[0] + app.userImageSpaceX + app.imgOffsetX
    facialFeatureY = faceTuple[1] + app.userImageSpaceY + app.imgOffsetY
    correctFacialFeatureTuple = (facialFeatureX, facialFeatureY, faceTuple[2], faceTuple[3])
    return correctFacialFeatureTuple

def getIndividualEyeCoordinates(app):
    eye1X = app.filteredEyes[0][0]
    eye2X = app.filteredEyes[0][1]

    if eye1X > eye2X :
        rightEyeCoords = [app.filteredEyes[0]]
        leftEyeCoords = [app.filteredEyes[1]]
    else:
        rightEyeCoords = [app.filteredEyes[1]]
        leftEyeCoords = [app.filteredEyes[0]]
    
    return leftEyeCoords, rightEyeCoords

def getFacialFeatureCentre(featureList):
    facialFeatureCoords = featureList[0]
    featureX, featureY, featureWidth, featureHeight = facialFeatureCoords
    return featureX + featureWidth/2, featureY + featureHeight/2

def getEyeSlope(leftEyeCentre, rightEyeCentre):
    leftEyeCentreX, leftEyeCentreY = leftEyeCentre
    rightEyeCentreX, rightEyeCentreY = rightEyeCentre
    slope = (rightEyeCentreY - leftEyeCentreY)/(rightEyeCentreX - leftEyeCentreX)
    return slope

def getFacialFeatureEdges(facialFeatureList, featureCentreCoords, slope):
    facialFeatureCoords = facialFeatureList[0]
    featureX, featureY, featureWidth, featureHeight = facialFeatureCoords
    featureCentreX, featureCentreY = featureCentreCoords
    dHeight = slope * featureWidth/2
    featureRightEdgeX, featureRightEdgeY = featureCentreX + featureWidth/2, featureCentreY + dHeight
    featureLeftEdgeX, featureLeftEdgeY = featureCentreX - featureWidth/2, featureCentreY - dHeight
    return (featureRightEdgeX, featureRightEdgeY, featureLeftEdgeX, featureLeftEdgeY)

def drawFacialFeatureHelpers(app):
    if app.imageGuide:
        if app.boxesGuide:
            drawFacialFeatureRectangles(app)
        if app.linesGuide:
            drawFacialFeatureLines(app)
        if app.dotsGuide:
            drawFacialFeaturePoints(app)
    if app.boxesGuide:
        drawFacialFeatureRectangles(app, app.OffsetXToDrawingCanvas)
    if app.linesGuide:
        drawFacialFeatureLines(app, app.OffsetXToDrawingCanvas)
    if app.dotsGuide:
        drawFacialFeaturePoints(app, app.OffsetXToDrawingCanvas)

def drawFacialFeatureRectangles(app, offsetX = 0):

    # draw the face rectangle
    faceX, faceY, faceL, faceH = app.faceCoordinates
    drawRect(faceX - offsetX, faceY, faceL, faceH, fill = None, borderWidth = 2, border = 'lightGreen')

    # Draw the nose rectangles
    noseCoordinates = app.filteredNose
    for (nx, ny, nw, nh) in noseCoordinates:
        drawRect(nx - offsetX, ny, nw, nh, fill = None, borderWidth = 2, border = 'orange')

    # Draw the mouth rectangles
    mouthCoordinates = app.filteredMouth
    for (mx, my, mw, mh) in mouthCoordinates:
        drawRect(mx - offsetX, my, mw, mh, fill = None, borderWidth = 2, border = 'red')

    # Draw the mouth rectangles
    mouthCoordinates = app.filteredLeftEye
    for (ex, ey, ew, eh) in mouthCoordinates:
        drawRect(ex - offsetX, ey, ew, eh, fill = None, borderWidth = 2, border = 'blue')
    
    # Draw the mouth rectangles
    mouthCoordinates = app.filteredRightEye
    for (ex, ey, ew, eh) in mouthCoordinates:
        drawRect(ex - offsetX, ey, ew, eh, fill = None, borderWidth = 2, border = 'blue')

def drawFacialFeatureLines(app, offsetX = 0):
    # Draw Eye Line
    drawLine(app.leftEyeCentre[0] - offsetX, app.leftEyeCentre[1], app.rightEyeCentre[0] - offsetX, app.rightEyeCentre[1])

    # Draw Left Eye Line
    drawLine(app.leftEyeRightEdgeX - offsetX, app.leftEyeRightEdgeY, app.leftEyeLeftEdgeX - offsetX, app.leftEyeLeftEdgeY, fill='green')

    # Draw Right Eye Line
    drawLine(app.rightEyeRightEdgeX - offsetX, app.rightEyeRightEdgeY, app.rightEyeLeftEdgeX - offsetX, app.rightEyeLeftEdgeY, fill='green')

    # Draw Nose Line
    drawLine(app.noseRightEdgeX - offsetX, app.noseRightEdgeY, app.noseLeftEdgeX - offsetX, app.noseLeftEdgeY, fill='green')

    # Draw Mouth Line
    drawLine(app.mouthRightEdgeX - offsetX, app.mouthRightEdgeY, app.mouthLeftEdgeX - offsetX, app.mouthLeftEdgeY, fill='green')

    # Draw Eye-Nose Line
    drawLine(app.bothEyesCentre[0] - offsetX, app.bothEyesCentre[1], app.noseCentre[0] - offsetX, app.noseCentre[1], fill='purple')

    # Draw Nose-Mouth Line
    drawLine(app.noseCentre[0] - offsetX, app.noseCentre[1], app.mouthCentre[0] - offsetX, app.mouthCentre[1], fill='purple')


def drawFacialFeaturePoints(app, offsetX = 0):

    # Draw Left Eye Centre
    leftEyeCentreX, leftEyeCentreY = app.leftEyeCentre
    drawCircle(leftEyeCentreX - offsetX, leftEyeCentreY, app.dotRadius, fill = 'lightGreen')

    # Draw Left Eye Centre
    rightEyeCentreX, rightEyeCentreY = app.rightEyeCentre
    drawCircle(rightEyeCentreX - offsetX, rightEyeCentreY, app.dotRadius, fill = 'lightGreen')

    # Draw Left Eye Centre
    noseCentreX, noseCentreY = app.noseCentre
    drawCircle(noseCentreX - offsetX, noseCentreY, app.dotRadius, fill = 'lightGreen')

    # Draw Left Eye Centre
    mouthCentreX, mouthCentreY = app.mouthCentre
    drawCircle(mouthCentreX - offsetX, mouthCentreY, app.dotRadius, fill = 'lightGreen')

    # Draw Left Eye Edges
    drawCircle(app.leftEyeRightEdgeX - offsetX, app.leftEyeRightEdgeY, app.dotRadius, fill='red')
    drawCircle(app.leftEyeLeftEdgeX - offsetX, app.leftEyeLeftEdgeY, app.dotRadius, fill='red')

    # Draw Right Eye Edges
    drawCircle(app.rightEyeRightEdgeX - offsetX, app.rightEyeRightEdgeY, app.dotRadius, fill='red')
    drawCircle(app.rightEyeLeftEdgeX - offsetX, app.rightEyeLeftEdgeY, app.dotRadius, fill='red')

    # Draw Nose Edges
    drawCircle(app.noseRightEdgeX - offsetX, app.noseRightEdgeY, app.dotRadius, fill='red')
    drawCircle(app.noseLeftEdgeX - offsetX, app.noseLeftEdgeY, app.dotRadius, fill='red')
    
    # Draw Mouth Edges
    drawCircle(app.mouthRightEdgeX - offsetX, app.mouthRightEdgeY, app.dotRadius, fill='red')
    drawCircle(app.mouthLeftEdgeX - offsetX, app.mouthLeftEdgeY, app.dotRadius, fill='red')

def doAllFaceCalculations(app):
    app.imgX, app.imgY = app.userImageSpaceX, app.userImageSpaceY
    app.imgWidth, app.imgHeight, app.resizeFactor = imageResizer(app.imgPath, app.userImageSpaceWidth, app.userImageSpaceHeight)
    app.grayImage =  grayScaleImage(app.imgPath)

    # Get face coordinates
    app.faceCoordinates = faceDetection(app, app.grayImage)
    faceX, faceY, faceL, faceH = app.faceCoordinates

    # Get all the eye regions
    app.eyeRegionCoordinates = eyeRegionDetection(app, app.grayImage, (faceX, faceY, faceL, faceH))
    eyeRegionX, eyeRegionY, eyeRegionL, eyeRegionH = app.eyeRegionCoordinates

    # Get all facial features
    app.eyeCoordinates = eyeDetection(app, app.grayImage, (faceX, faceY, faceL, faceH))
    app.realEyeCoordinates = rectIntersections(app.eyeCoordinates, [(eyeRegionX, eyeRegionY, eyeRegionL, eyeRegionH)])
    app.noseCoordinates = noseDetection(app, app.grayImage, (faceX, faceY, faceL, faceH))
    app.mouthCoordinates = mouthDetection(app, app.grayImage, (faceX, faceY, faceL, faceH))

    # Get all filtered facial feature
    app.filteredEyesBeforeCorrection = eyesFilter(app, app.realEyeCoordinates)
    app.filteredNoseBeforeCorrection = noseFilter(app, app.filteredEyesBeforeCorrection, app.noseCoordinates)
    app.filteredMouthBeforeCorrection = mouthFilter(app, app.filteredNoseBeforeCorrection, app.mouthCoordinates)

    app.imgCentreX = app.imgX + app.imgWidth/2
    app.imgCentreY = app.imgY + app.imgHeight/2

    app.userImageSpaceCentreX = app.userImageSpaceX + app.userImageSpaceWidth/2
    app.userImageSpaceCentreY = app.userImageSpaceY + app.userImageSpaceHeight/2

    app.imgOffsetX = app.userImageSpaceCentreX - app.imgCentreX
    app.imgOffsetY = app.userImageSpaceCentreY - app.imgCentreY

    app.imgX, app.imgY = app.imgX + app.imgOffsetX, app.imgY + app.imgOffsetY
    app.filteredEyes = offsetImageDetectionParts(app, app.filteredEyesBeforeCorrection)
    app.filteredNose = offsetImageDetectionParts(app, app.filteredNoseBeforeCorrection)
    app.filteredMouth  = offsetImageDetectionParts(app, app.filteredMouthBeforeCorrection)
    app.faceCoordinates = offsetImageFaceCoordinates(app, app.faceCoordinates)

    app.filteredLeftEye, app.filteredRightEye = getIndividualEyeCoordinates(app)

    app.leftEyeCentre = getFacialFeatureCentre(app.filteredLeftEye)
    app.rightEyeCentre = getFacialFeatureCentre(app.filteredRightEye)
    app.noseCentre = getFacialFeatureCentre(app.filteredNose)
    app.mouthCentre = getFacialFeatureCentre(app.filteredMouth)
    app.slope = getEyeSlope(app.leftEyeCentre, app.rightEyeCentre)

    app.leftEyeRightEdgeX, app.leftEyeRightEdgeY, app.leftEyeLeftEdgeX, app.leftEyeLeftEdgeY = getFacialFeatureEdges(app.filteredLeftEye, app.leftEyeCentre , app.slope)
    app.rightEyeRightEdgeX, app.rightEyeRightEdgeY, app.rightEyeLeftEdgeX, app.rightEyeLeftEdgeY = getFacialFeatureEdges(app.filteredRightEye, app.rightEyeCentre , app.slope)
    app.noseRightEdgeX, app.noseRightEdgeY, app.noseLeftEdgeX, app.noseLeftEdgeY = getFacialFeatureEdges(app.filteredNose, app.noseCentre , app.slope)
    app.mouthRightEdgeX, app.mouthRightEdgeY, app.mouthLeftEdgeX, app.mouthLeftEdgeY = getFacialFeatureEdges(app.filteredMouth, app.mouthCentre , app.slope)

    app.bothEyesCentre = ((app.leftEyeCentre[0]+app.rightEyeCentre[0])/2, (app.leftEyeCentre[1]+app.rightEyeCentre[1])/2)

    app.dotRadius = 4

    app.pilUserImageCanvas = createCanvas(int(app.imgWidth), int(app.imgHeight), app.canvasColor)
    image = Image.open(app.imgPath)
    image = image.resize((int(app.imgWidth), int(app.imgHeight)))
    app.pilUserImageCanvas.paste(image, (0, 0))

    app.canvasCentre = app.canvasX + app.canvasWidth/2
    app.OffsetXToDrawingCanvas = app.userImageSpaceCentreX - app.canvasCentre

def doAllFaceCalculationsForDrawingCanvas(app):
    app.imgX2, app.imgY2 = app.canvasX, app.canvasY
    app.imgWidth2, app.imgHeight2, app.resizeFactor2 = imageResizer(app.imgPath2, app.canvasWidth, app.canvasHeight)
    app.grayImage2 =  grayScaleImage(app.imgPath2)

    # Get face coordinates
    app.faceCoordinates2 = faceDetection(app, app.grayImage2)
    faceX, faceY, faceL, faceH = app.faceCoordinates2

    # Get all the eye regions
    app.eyeRegionCoordinates2 = eyeRegionDetection(app, app.grayImage2, (faceX, faceY, faceL, faceH))
    eyeRegionX, eyeRegionY, eyeRegionL, eyeRegionH = app.eyeRegionCoordinates2

    # Get all facial features
    app.eyeCoordinates2 = eyeDetection(app, app.grayImage2, (faceX, faceY, faceL, faceH))
    app.realEyeCoordinates2 = rectIntersections(app.eyeCoordinates2, [(eyeRegionX, eyeRegionY, eyeRegionL, eyeRegionH)])
    app.noseCoordinates2 = noseDetection(app, app.grayImage2, (faceX, faceY, faceL, faceH))
    app.mouthCoordinates2 = mouthDetection(app, app.grayImage2, (faceX, faceY, faceL, faceH))

    # Get all filtered facial feature
    app.filteredEyes2 = eyesFilter(app, app.realEyeCoordinates2)
    app.filteredNose2 = noseFilter(app, app.filteredEyes2, app.noseCoordinates2)
    app.filteredMouth2 = mouthFilter(app, app.filteredNose2, app.mouthCoordinates2)
    app.filteredLeftEye2, app.filteredRightEye2 = getIndividualEyeCoordinates(app)


################################################################################################################################
'''
Main
'''
################################################################################################################################
def main():
    runApp()

main()